/*!
The SPIR-V backend stores a built-in output array, such as the clip
distances, element by element rather than with one `OpStore` of the whole
array, which Qualcomm's Adreno X1 compiler needs (see
`Writer::builtin_output_array`).
*/

#![cfg(all(feature = "wgsl-in", spv_out))]

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
fn clip_distances_are_stored_element_by_element() {
    let module = spirv(
        "
        enable clip_distances;
        struct Out {
            @builtin(position) position: vec4<f32>,
            @builtin(clip_distances) clip: array<f32, 4>,
            @location(0) color: vec4<f32>,
        }
        fn planes(p: vec2<f32>) -> array<f32, 4> {
            return array<f32, 4>(p.x, 1.0 - p.x, p.y, 1.0 - p.y);
        }
        @vertex
        fn main(@builtin(vertex_index) index: u32) -> Out {
            var out: Out;
            out.position = vec4<f32>(f32(index));
            out.clip = planes(out.position.xy);
            out.color = vec4<f32>(1.0);
            return out;
        }
        ",
    );
    let arrays: Vec<u32> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::TypeArray)
        .filter_map(|i| i.result_id)
        .collect();
    let output = rspirv::dr::Operand::StorageClass(spirv::StorageClass::Output);
    let array_pointers: Vec<u32> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::TypePointer && i.operands[0] == output)
        .filter(|i| arrays.contains(&i.operands[1].unwrap_id_ref()))
        .filter_map(|i| i.result_id)
        .collect();
    let globals = || {
        module
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::Variable && i.operands[0] == output)
    };
    let outputs: Vec<u32> = globals().filter_map(|i| i.result_id).collect();
    // The clip distances.
    let array_outputs: Vec<u32> = globals()
        .filter(|i| array_pointers.contains(&i.result_type.unwrap()))
        .filter_map(|i| i.result_id)
        .collect();
    assert_eq!(array_outputs.len(), 1);
    let instructions: Vec<_> = module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .collect();
    let value_types: naga::FastHashMap<u32, u32> = instructions
        .iter()
        .filter_map(|i| Some((i.result_id?, i.result_type?)))
        .collect();
    let mut element_stores = 0;
    for instruction in &instructions {
        if instruction.class.opcode != Op::Store {
            continue;
        }
        let pointer = instruction.operands[0].unwrap_id_ref();
        let value = instruction.operands[1].unwrap_id_ref();
        if outputs.contains(&pointer) {
            assert!(
                !value_types
                    .get(&value)
                    .is_some_and(|ty| arrays.contains(ty)),
                "an array stored whole to an output: {instruction:?}"
            );
        }
        if instructions.iter().any(|i| {
            i.result_id == Some(pointer)
                && i.class.opcode == Op::AccessChain
                && array_outputs.contains(&i.operands[0].unwrap_id_ref())
        }) {
            element_stores += 1;
        }
    }
    assert_eq!(element_stores, 4);
}
