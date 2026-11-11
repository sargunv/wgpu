/*!
The SPIR-V backend never takes a struct out of a composite with one
`OpCompositeExtract`: it extracts the struct's members and puts them back
together, which Qualcomm's Adreno compilers need (see
`BlockContext::write_composite_extract`).
*/

#![cfg(all(feature = "wgsl-in", spv_out))]

use rspirv::dr::Operand;
use spirv::Op;

fn spirv(source: &str) -> rspirv::dr::Module {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let words =
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .unwrap();
    rspirv::dr::load_words(words).unwrap()
}

#[test]
fn struct_members_are_rebuilt() {
    let module = spirv(
        "
        struct Curve { params: vec4<f32>, kind: u32 }
        struct Size { kind: u32, curve: Curve, curves: array<Curve, 2> }
        @group(0) @binding(0) var<uniform> size: Size;

        fn factor(c: Curve) -> f32 {
            if c.kind == 2u { return c.params.x; }
            return c.params.y;
        }
        fn evaluate(s: Size) -> f32 {
            return factor(s.curve) + factor(s.curves[1]);
        }
        @vertex
        fn main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(evaluate(size));
        }
        ",
    );
    let structs: Vec<u32> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::TypeStruct)
        .filter_map(|i| i.result_id)
        .collect();
    let instructions = module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions);
    let mut constructs = 0;
    for instruction in instructions {
        let result_type = instruction.result_type.unwrap_or(0);
        match instruction.class.opcode {
            Op::CompositeExtract => assert!(
                !structs.contains(&result_type),
                "a struct extracted whole: {instruction:?}"
            ),
            Op::CompositeConstruct if structs.contains(&result_type) => {
                // `Curve(params, kind)`: one id per member.
                assert!(instruction
                    .operands
                    .iter()
                    .all(|o| matches!(o, Operand::IdRef(_))));
                constructs += 1;
            }
            _ => {}
        }
    }
    // `s.curve` and `s.curves[1]`.
    assert_eq!(constructs, 2);
}
