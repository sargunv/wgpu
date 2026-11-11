//! Struct values in functions split into their members.
//!
//! GLSL ES gives a struct's members their own precision, and a struct
//! variable none of its own. Some drivers still evaluate the members of a
//! struct-typed variable, argument or return value at the stage's default
//! precision, which in a fragment shader is `mediump`: the Adreno 619's
//! GLES driver keeps 16 bits of every integer member and a half float of
//! every float one. ANGLE, which every WebGL 2 shader on Android passes
//! through, drops the shader's `precision highp` statements, so such a
//! driver sees no default but `mediump` even where the source declares
//! `highp`. Naga writes entry point inputs as one struct, so every varying
//! a fragment shader reads would pass through one.
//!
//! [`flatten_structs`] rewrites a module so that no function holds a
//! struct value: struct locals become one local per member (recursively),
//! struct arguments one argument per member, struct results one `inout`
//! pointer argument per member, and an entry point's struct input one
//! input per member, each with its binding. Uniform and storage structs
//! stay as they are: their members are read in place. An entry point's
//! struct result is composed where it returns.

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::{
    arena::{Arena, Handle},
    proc::Emitter,
    valid::{FunctionInfo, ModuleInfo},
    Binding, Block, Expression, Function, FunctionArgument, LocalVariable, Module, Span, Statement,
    SwitchCase, Type, TypeInner,
};

/// Why a module couldn't be flattened: something the pass doesn't handle.
#[derive(Clone, Debug, thiserror::Error)]
#[error("can't flatten struct values: {0}")]
pub struct FlattenError(&'static str);

/// One non-struct part of a struct type, in member order.
#[derive(Clone, Debug)]
struct Leaf {
    ty: Handle<Type>,
    /// Member names joined by `_`.
    name: String,
    /// The top-level member's binding.
    binding: Option<Binding>,
}

/// A flattenable struct type's leaves, and each top-level member's range
/// of them.
#[derive(Clone, Debug)]
struct Layout {
    leaves: Vec<Leaf>,
    members: Vec<core::ops::Range<usize>>,
}

/// An old expression's value in the new function.
#[derive(Clone, Debug)]
enum Mapped {
    One(Handle<Expression>),
    /// A flattenable struct value or pointer, as its leaves (or pointers
    /// to them).
    Leaves(Vec<Handle<Expression>>),
}

/// How a function's signature changed.
#[derive(Clone, Debug, Default)]
struct Signature {
    /// Per old argument, the range of new arguments it became.
    arguments: Vec<core::ops::Range<u32>>,
    /// The result's struct type, if it was split into out arguments.
    result: Option<Handle<Type>>,
}

struct Pass<'a> {
    module: &'a Module,
    layouts: Vec<Option<Layout>>,
    /// `ptr<function, T>` by `T`, for out arguments.
    pointers: Vec<(Handle<Type>, Handle<Type>)>,
    signatures: Vec<Signature>,
}

/// Returns `module` with no struct value held in a function (see the
/// module docs). The result needs validating again before it is written.
pub fn flatten_structs(module: &Module, info: &ModuleInfo) -> Result<Module, FlattenError> {
    let mut pass = Pass {
        module,
        layouts: Vec::new(),
        pointers: Vec::new(),
        signatures: Vec::new(),
    };
    pass.find_layouts();
    let mut out = module.clone();
    // Signatures first: calls need their callees'.
    for (_, function) in module.functions.iter() {
        let signature = pass.signature(function, false);
        if let Some(ty) = signature.result {
            for leaf in pass.layout(ty).expect("splits").leaves.clone() {
                if !pass.pointers.iter().any(|&(base, _)| base == leaf.ty) {
                    let pointer = out.types.insert(
                        Type {
                            name: None,
                            inner: TypeInner::Pointer {
                                base: leaf.ty,
                                space: crate::AddressSpace::Function,
                            },
                        },
                        Span::UNDEFINED,
                    );
                    pass.pointers.push((leaf.ty, pointer));
                }
            }
        }
        pass.signatures.push(signature);
    }
    for (handle, function) in module.functions.iter() {
        let signature = pass.signatures[handle.index()].clone();
        out.functions[handle] = pass.function(function, &info[handle], &signature, false)?;
    }
    for (index, entry) in module.entry_points.iter().enumerate() {
        let signature = pass.signature(&entry.function, true);
        out.entry_points[index].function = pass.function(
            &entry.function,
            info.get_entry_point(index),
            &signature,
            true,
        )?;
    }
    Ok(out)
}

impl Pass<'_> {
    /// Which struct types split: those whose members are all leaves or
    /// splittable structs, but for the predeclared ones (`modf`'s,
    /// atomics') the backends write themselves.
    fn find_layouts(&mut self) {
        let types = &self.module.types;
        self.layouts = vec![None; types.len()];
        let special: Vec<_> = (self.module.special_types.predeclared_types.values())
            .copied()
            .collect();
        // Members come before the structs holding them in the arena.
        for (handle, ty) in types.iter() {
            let TypeInner::Struct { ref members, .. } = ty.inner else {
                continue;
            };
            if special.contains(&handle) {
                continue;
            }
            let mut layout = Layout {
                leaves: Vec::new(),
                members: Vec::new(),
            };
            let mut ok = true;
            for (i, member) in members.iter().enumerate() {
                let name = member.name.clone().unwrap_or_else(|| format!("m{i}"));
                let start = layout.leaves.len();
                match types[member.ty].inner {
                    TypeInner::Struct { .. } => match self.layouts[member.ty.index()] {
                        Some(ref inner) => {
                            for leaf in &inner.leaves {
                                layout.leaves.push(Leaf {
                                    ty: leaf.ty,
                                    name: format!("{name}_{}", leaf.name),
                                    binding: member.binding.clone(),
                                });
                            }
                        }
                        None => ok = false,
                    },
                    TypeInner::Array {
                        size: crate::ArraySize::Dynamic,
                        ..
                    } => ok = false,
                    _ => layout.leaves.push(Leaf {
                        ty: member.ty,
                        name,
                        binding: member.binding.clone(),
                    }),
                }
                layout.members.push(start..layout.leaves.len());
            }
            if ok && !layout.leaves.is_empty() {
                self.layouts[handle.index()] = Some(layout);
            }
        }
    }

    fn layout(&self, ty: Handle<Type>) -> Option<&Layout> {
        self.layouts.get(ty.index())?.as_ref()
    }

    fn splits(&self, ty: Handle<Type>) -> bool {
        self.layout(ty).is_some()
    }

    fn pointer_to(&self, base: Handle<Type>) -> Handle<Type> {
        let &(_, pointer) = (self.pointers.iter())
            .find(|&&(b, _)| b == base)
            .expect("made for the signature");
        pointer
    }

    fn signature(&mut self, function: &Function, entry: bool) -> Signature {
        let mut signature = Signature::default();
        let mut next = 0u32;
        for argument in &function.arguments {
            let count = self.layout(argument.ty).map_or(1, |l| l.leaves.len()) as u32;
            signature.arguments.push(next..next + count);
            next += count;
        }
        if !entry {
            if let Some(ref result) = function.result {
                if self.splits(result.ty) {
                    signature.result = Some(result.ty);
                }
            }
        }
        signature
    }

    fn function(
        &mut self,
        old: &Function,
        info: &FunctionInfo,
        signature: &Signature,
        entry: bool,
    ) -> Result<Function, FlattenError> {
        let mut new = Function {
            name: old.name.clone(),
            arguments: Vec::new(),
            result: old.result.clone(),
            local_variables: Arena::new(),
            expressions: Arena::new(),
            named_expressions: Default::default(),
            body: Block::new(),
            diagnostic_filter_leaf: old.diagnostic_filter_leaf,
        };
        for argument in &old.arguments {
            let name = argument.name.clone().unwrap_or_else(|| "arg".to_string());
            match self.layout(argument.ty).cloned() {
                Some(layout) => {
                    for leaf in layout.leaves {
                        new.arguments.push(FunctionArgument {
                            name: Some(format!("{name}_{}", leaf.name)),
                            ty: leaf.ty,
                            binding: if entry { leaf.binding } else { None },
                        });
                    }
                }
                None => new.arguments.push(argument.clone()),
            }
        }
        let mut out_arguments = Vec::new();
        if let Some(ty) = signature.result {
            let layout = self.layout(ty).cloned().expect("splits");
            for leaf in layout.leaves {
                out_arguments.push(new.arguments.len() as u32);
                new.arguments.push(FunctionArgument {
                    name: Some(format!("result_{}", leaf.name)),
                    ty: self.pointer_to(leaf.ty),
                    binding: None,
                });
            }
            new.result = None;
        }
        let mut f = FunctionPass {
            pass: self,
            old,
            info,
            new,
            map: vec![None; old.expressions.len()],
            locals: Vec::new(),
            emitter: Emitter::default(),
            emitted: vec![false; old.expressions.len()],
            out_arguments,
            signature,
        };
        f.mark_emitted(&old.body);
        let mut body = Block::new();
        let inits = f.locals();
        f.block(&old.body, &mut body)?;
        f.flush(&mut body);
        f.initialize(inits, &mut body)?;
        f.flush(&mut body);
        f.new.body = body;
        Ok(f.new)
    }
}

struct FunctionPass<'a, 'p> {
    pass: &'p mut Pass<'a>,
    old: &'p Function,
    info: &'p FunctionInfo,
    new: Function,
    map: Vec<Option<Mapped>>,
    /// Per old local, the new local(s).
    locals: Vec<Vec<Handle<LocalVariable>>>,
    emitter: Emitter,
    /// Old expressions some `Emit` covers.
    emitted: Vec<bool>,
    out_arguments: Vec<u32>,
    signature: &'p Signature,
}

impl FunctionPass<'_, '_> {
    fn mark_emitted(&mut self, block: &Block) {
        for statement in block.iter() {
            match *statement {
                Statement::Emit(ref range) => {
                    for h in range.clone() {
                        self.emitted[h.index()] = true;
                    }
                }
                Statement::Block(ref b) => self.mark_emitted(b),
                Statement::If {
                    ref accept,
                    ref reject,
                    ..
                } => {
                    self.mark_emitted(accept);
                    self.mark_emitted(reject);
                }
                Statement::Switch { ref cases, .. } => {
                    for case in cases {
                        self.mark_emitted(&case.body);
                    }
                }
                Statement::Loop {
                    ref body,
                    ref continuing,
                    ..
                } => {
                    self.mark_emitted(body);
                    self.mark_emitted(continuing);
                }
                _ => {}
            }
        }
    }

    /// Ends the running `Emit`, if any, into `out`.
    fn flush(&mut self, out: &mut Block) {
        if self.emitter.is_running() {
            if let Some((statement, span)) = self.emitter.finish(&self.new.expressions) {
                out.push(statement, span);
            }
        }
    }

    /// Appends `expr`: in an `Emit` if `emit` and it needs one, otherwise
    /// outside them.
    fn add(
        &mut self,
        expr: Expression,
        span: Span,
        emit: bool,
        out: &mut Block,
    ) -> Handle<Expression> {
        let emit = emit && !expr.needs_pre_emit() && !is_result(&expr);
        if emit {
            if !self.emitter.is_running() {
                self.emitter.start(&self.new.expressions);
            }
        } else {
            self.flush(out);
        }
        self.new.expressions.append(expr, span)
    }

    /// The old expression's struct type, if it's a splittable struct value.
    fn struct_value(&self, old: Handle<Expression>) -> Option<Handle<Type>> {
        let ty = self.info[old].ty.handle()?;
        self.pass.splits(ty).then_some(ty)
    }

    /// The pointee, if the old expression points to a splittable struct.
    fn struct_pointee(&self, old: Handle<Expression>) -> Option<Handle<Type>> {
        let types = &self.pass.module.types;
        let base = match *self.info[old].ty.inner_with(types) {
            TypeInner::Pointer { base, .. } => base,
            _ => return None,
        };
        self.pass.splits(base).then_some(base)
    }

    /// Makes the new locals, returning each initializer and the locals it
    /// initializes, for [`Self::initialize`] once the body is translated.
    #[allow(clippy::type_complexity)]
    fn locals(&mut self) -> Vec<(Vec<Handle<LocalVariable>>, Handle<Expression>)> {
        let mut inits = Vec::new();
        for (handle, local) in self.old.local_variables.iter() {
            let name = local.name.clone().unwrap_or_else(|| "local".to_string());
            match self.pass.layout(local.ty).cloned() {
                Some(layout) => {
                    let made: Vec<_> = (layout.leaves.iter())
                        .map(|leaf| {
                            self.new.local_variables.append(
                                LocalVariable {
                                    name: Some(format!("{name}_{}", leaf.name)),
                                    ty: leaf.ty,
                                    init: None,
                                },
                                self.old.local_variables.get_span(handle),
                            )
                        })
                        .collect();
                    if let Some(init) = local.init {
                        inits.push((made.clone(), init));
                    }
                    self.locals.push(made);
                }
                None => {
                    let made = self.new.local_variables.append(
                        LocalVariable {
                            name: local.name.clone(),
                            ty: local.ty,
                            init: None,
                        },
                        self.old.local_variables.get_span(handle),
                    );
                    if let Some(init) = local.init {
                        inits.push((vec![made], init));
                    }
                    self.locals.push(vec![made]);
                }
            }
        }
        inits
    }

    /// Initializes the new locals: an initializer is a constant expression,
    /// which may be one the body emits, so its leaves are known once the
    /// body is translated.
    #[allow(clippy::type_complexity)]
    fn initialize(
        &mut self,
        inits: Vec<(Vec<Handle<LocalVariable>>, Handle<Expression>)>,
        out: &mut Block,
    ) -> Result<(), FlattenError> {
        for (made, init) in inits {
            let leaves = match self.get(init, out)? {
                Mapped::Leaves(leaves) => leaves,
                Mapped::One(h) if made.len() == 1 => vec![h],
                Mapped::One(_) => return Err(FlattenError("a struct local's initializer")),
            };
            for (local, value) in made.iter().zip(leaves) {
                self.new.local_variables.get_mut(*local).init = Some(value);
            }
        }
        Ok(())
    }

    /// The new value of old expression `old`, translating it first if it
    /// needs no `Emit` (it may be used before any statement reaches it).
    fn get(&mut self, old: Handle<Expression>, out: &mut Block) -> Result<Mapped, FlattenError> {
        if let Some(ref mapped) = self.map[old.index()] {
            return Ok(mapped.clone());
        }
        if self.emitted[old.index()] {
            return Err(FlattenError("an expression used before it's emitted"));
        }
        self.translate(old, out)?;
        Ok(self.map[old.index()].clone().expect("translated"))
    }

    /// `get`, as a single value: a split struct composed again.
    fn value(
        &mut self,
        old: Handle<Expression>,
        out: &mut Block,
    ) -> Result<Handle<Expression>, FlattenError> {
        match self.get(old, out)? {
            Mapped::One(h) => Ok(h),
            Mapped::Leaves(leaves) => {
                let ty =
                    (self.struct_value(old)).ok_or(FlattenError("a split pointer used whole"))?;
                let span = self.old.expressions.get_span(old);
                let emit = self.emitted[old.index()];
                Ok(self.compose(ty, &leaves, span, emit, out))
            }
        }
    }

    /// Composes struct `ty` from its leaves.
    fn compose(
        &mut self,
        ty: Handle<Type>,
        leaves: &[Handle<Expression>],
        span: Span,
        emit: bool,
        out: &mut Block,
    ) -> Handle<Expression> {
        let layout = self.pass.layout(ty).cloned().expect("splits");
        let TypeInner::Struct { ref members, .. } = self.pass.module.types[ty].inner else {
            unreachable!()
        };
        let members = members.clone();
        let mut components = Vec::new();
        for (member, range) in members.iter().zip(layout.members) {
            if self.pass.splits(member.ty) {
                components.push(self.compose(member.ty, &leaves[range], span, emit, out));
            } else {
                components.push(leaves[range.start]);
            }
        }
        self.add(Expression::Compose { ty, components }, span, emit, out)
    }

    /// The leaves of `h`, a struct value (or pointer) of type `ty` not
    /// split: an `AccessIndex` chain to each.
    fn decompose(
        &mut self,
        h: Handle<Expression>,
        ty: Handle<Type>,
        span: Span,
        out: &mut Block,
    ) -> Vec<Handle<Expression>> {
        let layout = self.pass.layout(ty).cloned().expect("splits");
        let TypeInner::Struct { ref members, .. } = self.pass.module.types[ty].inner else {
            unreachable!()
        };
        let members = members.clone();
        let mut leaves = Vec::with_capacity(layout.leaves.len());
        for (i, (member, _)) in members.iter().zip(layout.members).enumerate() {
            let access = self.add(
                Expression::AccessIndex {
                    base: h,
                    index: i as u32,
                },
                span,
                true,
                out,
            );
            if self.pass.splits(member.ty) {
                leaves.extend(self.decompose(access, member.ty, span, out));
            } else {
                leaves.push(access);
            }
        }
        leaves
    }

    /// `get`, as leaves: `ty` the struct (or pointee) type.
    fn leaves(
        &mut self,
        old: Handle<Expression>,
        ty: Handle<Type>,
        out: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, FlattenError> {
        Ok(match self.get(old, out)? {
            Mapped::Leaves(leaves) => leaves,
            Mapped::One(h) => {
                let span = self.old.expressions.get_span(old);
                self.decompose(h, ty, span, out)
            }
        })
    }

    fn translate(&mut self, old: Handle<Expression>, out: &mut Block) -> Result<(), FlattenError> {
        let span = self.old.expressions.get_span(old);
        let emit = self.emitted[old.index()];
        let fresh = self.new.expressions.len();
        let mapped = match self.old.expressions[old] {
            Expression::FunctionArgument(i) => {
                let range = self.signature.arguments[i as usize].clone();
                let args: Vec<_> = range
                    .map(|j| self.add(Expression::FunctionArgument(j), span, false, out))
                    .collect();
                if self.struct_value(old).is_some() {
                    Mapped::Leaves(args)
                } else {
                    Mapped::One(args[0])
                }
            }
            Expression::LocalVariable(local) => {
                let made = self.locals[local.index()].clone();
                let pointers: Vec<_> = made
                    .into_iter()
                    .map(|l| self.add(Expression::LocalVariable(l), span, false, out))
                    .collect();
                if self.struct_pointee(old).is_some() {
                    Mapped::Leaves(pointers)
                } else {
                    Mapped::One(pointers[0])
                }
            }
            Expression::ZeroValue(ty) if self.pass.splits(ty) => {
                let layout = self.pass.layout(ty).cloned().expect("splits");
                Mapped::Leaves(
                    (layout.leaves.iter())
                        .map(|leaf| self.add(Expression::ZeroValue(leaf.ty), span, false, out))
                        .collect(),
                )
            }
            Expression::AccessIndex { base, index } => match self.get(base, out)? {
                Mapped::Leaves(leaves) => {
                    let ty = (self.struct_value(base))
                        .or_else(|| self.struct_pointee(base))
                        .expect("split");
                    let range =
                        self.pass.layout(ty).expect("splits").members[index as usize].clone();
                    let TypeInner::Struct { ref members, .. } = self.pass.module.types[ty].inner
                    else {
                        unreachable!()
                    };
                    if self.pass.splits(members[index as usize].ty) {
                        Mapped::Leaves(leaves[range].to_vec())
                    } else {
                        Mapped::One(leaves[range.start])
                    }
                }
                Mapped::One(base) => {
                    Mapped::One(self.add(Expression::AccessIndex { base, index }, span, emit, out))
                }
            },
            Expression::Load { pointer } => match self.struct_value(old) {
                Some(ty) => {
                    let pointers = self.leaves(pointer, ty, out)?;
                    Mapped::Leaves(
                        pointers
                            .into_iter()
                            .map(|pointer| self.add(Expression::Load { pointer }, span, emit, out))
                            .collect(),
                    )
                }
                None => {
                    let pointer = self.value(pointer, out)?;
                    Mapped::One(self.add(Expression::Load { pointer }, span, emit, out))
                }
            },
            Expression::Compose { ty, ref components } => {
                let components = components.clone();
                if self.pass.splits(ty) {
                    let layout = self.pass.layout(ty).cloned().expect("splits");
                    let mut leaves = Vec::with_capacity(layout.leaves.len());
                    for component in components {
                        match self.struct_value(component) {
                            Some(cty) => leaves.extend(self.leaves(component, cty, out)?),
                            None => leaves.push(self.value(component, out)?),
                        }
                    }
                    Mapped::Leaves(leaves)
                } else {
                    let mut new = Vec::with_capacity(components.len());
                    for component in components {
                        new.push(self.value(component, out)?);
                    }
                    Mapped::One(self.add(
                        Expression::Compose {
                            ty,
                            components: new,
                        },
                        span,
                        emit,
                        out,
                    ))
                }
            }
            Expression::CallResult(_) => return Err(FlattenError("a call result before its call")),
            ref other => {
                let mut expr = other.clone();
                let mut operands = Vec::new();
                for_each_operand(&mut expr, &mut |h| operands.push(*h)).map_err(FlattenError)?;
                let mut new_operands = Vec::with_capacity(operands.len());
                for h in operands {
                    new_operands.push(self.value(h, out)?);
                }
                let mut next = new_operands.into_iter();
                for_each_operand(&mut expr, &mut |h| *h = next.next().expect("same operands"))
                    .map_err(FlattenError)?;
                let h = self.add(expr, span, emit, out);
                match self.struct_value(old) {
                    Some(ty) => Mapped::Leaves(self.decompose(h, ty, span, out)),
                    None => Mapped::One(h),
                }
            }
        };
        self.name(old, &mapped, fresh);
        self.map[old.index()] = Some(mapped);
        Ok(())
    }

    /// Names the new expressions made for `old`, as it was named.
    fn name(&mut self, old: Handle<Expression>, mapped: &Mapped, fresh: usize) {
        let Some(name) = self.old.named_expressions.get(&old) else {
            return;
        };
        match *mapped {
            Mapped::One(h) if h.index() >= fresh => {
                self.new.named_expressions.insert(h, name.clone());
            }
            Mapped::Leaves(ref leaves) => {
                let ty = self.struct_value(old);
                let names = ty
                    .and_then(|ty| self.pass.layout(ty))
                    .map(|l| l.leaves.clone());
                if let Some(names) = names {
                    for (h, leaf) in leaves.iter().zip(names) {
                        if h.index() >= fresh {
                            (self.new.named_expressions)
                                .insert(*h, format!("{name}_{}", leaf.name));
                        }
                    }
                }
            }
            Mapped::One(_) => {}
        }
    }

    fn block(&mut self, old: &Block, out: &mut Block) -> Result<(), FlattenError> {
        for (statement, &span) in old.span_iter() {
            self.statement(statement, span, out)?;
        }
        Ok(())
    }

    /// A nested block: the caller has ended its own `Emit`.
    fn sub_block(&mut self, old: &Block) -> Result<Block, FlattenError> {
        debug_assert!(!self.emitter.is_running());
        let mut block = Block::new();
        self.block(old, &mut block)?;
        self.flush(&mut block);
        Ok(block)
    }

    fn statement(
        &mut self,
        statement: &Statement,
        span: Span,
        out: &mut Block,
    ) -> Result<(), FlattenError> {
        match *statement {
            Statement::Emit(ref range) => {
                for h in range.clone() {
                    self.translate(h, out)?;
                }
            }
            Statement::Block(ref block) => {
                self.flush(out);
                let block = self.sub_block(block)?;
                self.flush(out);
                out.push(Statement::Block(block), span);
            }
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                let condition = self.value(condition, out)?;
                self.flush(out);
                let accept = self.sub_block(accept)?;
                let reject = self.sub_block(reject)?;
                self.flush(out);
                out.push(
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    span,
                );
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                let selector = self.value(selector, out)?;
                self.flush(out);
                let mut new = Vec::with_capacity(cases.len());
                for case in cases {
                    new.push(SwitchCase {
                        value: case.value,
                        body: self.sub_block(&case.body)?,
                        fall_through: case.fall_through,
                    });
                }
                self.flush(out);
                out.push(
                    Statement::Switch {
                        selector,
                        cases: new,
                    },
                    span,
                );
            }
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                self.flush(out);
                let body = self.sub_block(body)?;
                let mut continuing_block = Block::new();
                self.block(continuing, &mut continuing_block)?;
                let break_if = match break_if {
                    Some(h) => Some(self.value(h, &mut continuing_block)?),
                    None => None,
                };
                self.flush(&mut continuing_block);
                self.flush(out);
                out.push(
                    Statement::Loop {
                        body,
                        continuing: continuing_block,
                        break_if,
                    },
                    span,
                );
            }
            Statement::Break
            | Statement::Continue
            | Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_) => {
                self.flush(out);
                out.push(statement.clone(), span);
            }
            Statement::Return { value } => match value {
                Some(value) if !self.out_arguments.is_empty() => {
                    let ty = self
                        .struct_value(value)
                        .ok_or(FlattenError("a split result"))?;
                    let leaves = self.leaves(value, ty, out)?;
                    let outs = self.out_arguments.clone();
                    let mut pointers = Vec::new();
                    for &arg in &outs {
                        pointers.push(self.add(
                            Expression::FunctionArgument(arg),
                            span,
                            false,
                            out,
                        ));
                    }
                    self.flush(out);
                    for (pointer, value) in pointers.into_iter().zip(leaves) {
                        out.push(Statement::Store { pointer, value }, span);
                    }
                    out.push(Statement::Return { value: None }, span);
                }
                Some(value) => {
                    let value = self.value(value, out)?;
                    self.flush(out);
                    out.push(Statement::Return { value: Some(value) }, span);
                }
                None => {
                    self.flush(out);
                    out.push(Statement::Return { value: None }, span);
                }
            },
            Statement::Store { pointer, value } => match self.struct_pointee(pointer) {
                Some(ty) => {
                    let pointers = self.leaves(pointer, ty, out)?;
                    let values = self.leaves(value, ty, out)?;
                    self.flush(out);
                    for (pointer, value) in pointers.into_iter().zip(values) {
                        out.push(Statement::Store { pointer, value }, span);
                    }
                }
                None => {
                    let pointer = self.value(pointer, out)?;
                    let value = self.value(value, out)?;
                    self.flush(out);
                    out.push(Statement::Store { pointer, value }, span);
                }
            },
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                let callee = &self.pass.module.functions[function];
                let signature = self.pass.signatures[function.index()].clone();
                let mut new = Vec::new();
                for (argument, param) in arguments.iter().zip(&callee.arguments) {
                    if self.pass.splits(param.ty) {
                        new.extend(self.leaves(*argument, param.ty, out)?);
                    } else {
                        new.push(self.value(*argument, out)?);
                    }
                }
                match (signature.result, result) {
                    (Some(ty), result) => {
                        let layout = self.pass.layout(ty).cloned().expect("splits");
                        let name = callee.name.clone().unwrap_or_else(|| "call".to_string());
                        let mut temps = Vec::new();
                        for leaf in &layout.leaves {
                            let local = self.new.local_variables.append(
                                LocalVariable {
                                    name: Some(format!("{name}_{}", leaf.name)),
                                    ty: leaf.ty,
                                    init: None,
                                },
                                span,
                            );
                            let pointer =
                                self.add(Expression::LocalVariable(local), span, false, out);
                            temps.push(pointer);
                            new.push(pointer);
                        }
                        self.flush(out);
                        out.push(
                            Statement::Call {
                                function,
                                arguments: new,
                                result: None,
                            },
                            span,
                        );
                        if let Some(result) = result {
                            let fresh = self.new.expressions.len();
                            let loads: Vec<_> = temps
                                .into_iter()
                                .map(|pointer| {
                                    self.add(Expression::Load { pointer }, span, true, out)
                                })
                                .collect();
                            let mapped = Mapped::Leaves(loads);
                            self.name(result, &mapped, fresh);
                            self.map[result.index()] = Some(mapped);
                        }
                    }
                    (None, result) => {
                        self.flush(out);
                        let made = result.map(|result| {
                            let h = self.add(Expression::CallResult(function), span, false, out);
                            self.map[result.index()] = Some(Mapped::One(h));
                            h
                        });
                        out.push(
                            Statement::Call {
                                function,
                                arguments: new,
                                result: made,
                            },
                            span,
                        );
                    }
                }
            }
            _ => return Err(FlattenError("a statement it doesn't handle")),
        }
        Ok(())
    }
}

/// Whether a statement makes the expression, so no `Emit` covers it.
const fn is_result(expr: &Expression) -> bool {
    matches!(
        *expr,
        Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::RayQueryProceedResult
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. }
    )
}

/// Calls `f` on each expression operand of `expr`, in a fixed order.
fn for_each_operand(
    expr: &mut Expression,
    f: &mut dyn FnMut(&mut Handle<Expression>),
) -> Result<(), &'static str> {
    use crate::{ImageQuery, SampleLevel};
    let opt = |h: &mut Option<Handle<Expression>>, f: &mut dyn FnMut(&mut Handle<Expression>)| {
        if let Some(h) = h.as_mut() {
            f(h);
        }
    };
    match *expr {
        Expression::Literal(_)
        | Expression::Constant(_)
        | Expression::Override(_)
        | Expression::ZeroValue(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_) => {}
        Expression::Compose {
            ref mut components, ..
        } => components.iter_mut().for_each(f),
        Expression::Access {
            ref mut base,
            ref mut index,
        } => {
            f(base);
            f(index);
        }
        Expression::AccessIndex { ref mut base, .. } => f(base),
        Expression::Splat { ref mut value, .. } => f(value),
        Expression::Swizzle { ref mut vector, .. } => f(vector),
        Expression::Load { ref mut pointer } => f(pointer),
        Expression::ImageSample {
            ref mut image,
            ref mut sampler,
            ref mut coordinate,
            ref mut array_index,
            ref mut offset,
            ref mut level,
            ref mut depth_ref,
            ..
        } => {
            f(image);
            f(sampler);
            f(coordinate);
            opt(array_index, f);
            opt(offset, f);
            match *level {
                SampleLevel::Auto | SampleLevel::Zero => {}
                SampleLevel::Exact(ref mut h) | SampleLevel::Bias(ref mut h) => f(h),
                SampleLevel::Gradient {
                    ref mut x,
                    ref mut y,
                } => {
                    f(x);
                    f(y);
                }
            }
            opt(depth_ref, f);
        }
        Expression::ImageLoad {
            ref mut image,
            ref mut coordinate,
            ref mut array_index,
            ref mut sample,
            ref mut level,
        } => {
            f(image);
            f(coordinate);
            opt(array_index, f);
            opt(sample, f);
            opt(level, f);
        }
        Expression::ImageQuery {
            ref mut image,
            ref mut query,
        } => {
            f(image);
            if let ImageQuery::Size {
                level: Some(ref mut level),
            } = *query
            {
                f(level);
            }
        }
        Expression::Unary { ref mut expr, .. } => f(expr),
        Expression::Binary {
            ref mut left,
            ref mut right,
            ..
        } => {
            f(left);
            f(right);
        }
        Expression::Select {
            ref mut condition,
            ref mut accept,
            ref mut reject,
        } => {
            f(condition);
            f(accept);
            f(reject);
        }
        Expression::Derivative { ref mut expr, .. } => f(expr),
        Expression::Relational {
            ref mut argument, ..
        } => f(argument),
        Expression::Math {
            ref mut arg,
            ref mut arg1,
            ref mut arg2,
            ref mut arg3,
            ..
        } => {
            f(arg);
            opt(arg1, f);
            opt(arg2, f);
            opt(arg3, f);
        }
        Expression::As { ref mut expr, .. } => f(expr),
        Expression::ArrayLength(ref mut expr) => f(expr),
        _ => return Err("an expression it doesn't handle"),
    }
    Ok(())
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use alloc::string::String;

    use crate::back::glsl;
    use crate::valid::{Capabilities, ValidationFlags, Validator};

    /// A fragment shader passing structs in and out of functions, and
    /// taking its inputs as one, written with no struct-typed variable.
    #[test]
    fn no_struct_values_in_fragment_glsl() {
        let source = "
            struct Inner { a: u32, b: array<vec4<f32>, 2> }
            struct Outer { inner: Inner, c: vec2<f32> }
            struct In {
                @builtin(position) position: vec4<f32>,
                @location(0) @interpolate(flat) word: u32,
                @location(1) uv: vec2<f32>,
            }
            @group(0) @binding(0) var<uniform> u: Outer;
            fn make(word: u32, uv: vec2<f32>) -> Outer {
                var o: Outer;
                o.inner.a = word;
                o.inner.b = array<vec4<f32>, 2>(vec4<f32>(uv, 0.0, 0.0), vec4<f32>(uv.yx, 0.0, 0.0));
                o.c = uv;
                return o;
            }
            fn use_it(o: Outer) -> f32 {
                return f32(o.inner.a >> 16u) + o.inner.b[1].x + o.c.x;
            }
            @fragment
            fn main(in: In) -> @location(0) vec4<f32> {
                let o = make(in.word, in.uv);
                let copy = o;
                return vec4<f32>(use_it(copy), use_it(u), in.position.x, 1.0);
            }
        ";
        let module = crate::front::wgsl::parse_str(source).unwrap();
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap();
        let flat = super::flatten_structs(&module, &info).unwrap();
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&flat)
            .unwrap();
        let mut out = String::new();
        let options = glsl::Options {
            version: glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        let pipeline = glsl::PipelineOptions {
            shader_stage: crate::ShaderStage::Fragment,
            entry_point: "main".into(),
            multiview: None,
        };
        glsl::Writer::new(
            &mut out,
            &flat,
            &info,
            &options,
            &pipeline,
            crate::proc::BoundsCheckPolicies::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        // The struct definitions and the uniform block hold them; no
        // function does.
        let functions = &out[out.find("\nvoid ").unwrap()..];
        for name in ["Inner", "Outer", "In"] {
            assert!(
                !functions.contains(&alloc::format!("{name} ")),
                "{name} values in:\n{out}"
            );
        }
    }
}
