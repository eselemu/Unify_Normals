import sys
from mesh_processing import MeshProcessor

sys.setrecursionlimit(12000)

def main():
    processor = MeshProcessor()
    filename = 'unify_normals_01'
    filename = 's03'
    input_file = './Input/' + filename + '.obj'
    processor.read_obj_file(input_file)
    processor.convert_to_heds()
    processor.calculate_face_normals()
    processor.calculate_vertex_normals()

    # Detect components and process each
    components = processor.get_connected_components()
    print(f"Processing {len(components)} components")

    if len(components) == 1:
        processor.unify_normals()
        processor.correct_normals()
    else:
        for i, component in enumerate(components):
            print(f"\nProcessing Component {i}")
            processor.unify_normals_component(component)
            processor.calculate_face_normals()
            processor.calculate_vertex_normals()
            processor.correct_normals_for_component(component, outwards=True)

    # Optionally color the mesh after processing
    processor.color_faces_by_orientation()

    output_file = './Output/output_' + filename + '.obj'
    processor.write_obj_file(output_file)
    print(f"Colored mesh saved to {output_file}")

if __name__ == "__main__":
    main()
