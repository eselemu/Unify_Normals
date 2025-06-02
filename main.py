import sys
from mesh_processing import MeshProcessor

sys.setrecursionlimit(12000)

def main():
    processor = MeshProcessor()
    filename = 'unify_normals_03'  # Input file name
    input_file = './Input/' + filename + '.obj'
    
    # Load and process mesh
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
            flip = False
            processor.unify_normals_component(component)
            processor.calculate_face_normals()
            processor.calculate_vertex_normals()
            other_faces = set(face for comp in components if comp is not component for face in comp)
            
            # Check orientation using raycasting
            self_collision, self_diff = processor.evaluate_orientation_by_raycast(component)
            collision, diff = processor.evaluate_orientation_by_raycast(component, other_faces=other_faces)
            
            if (self_collision and self_diff < 0 and diff != 0) or (self_collision and self_diff > 0 and diff == 0):
                flip = True
            elif diff < 0:
                flip = True
            
            if flip:
                print(f"Flipping component {i} based on raycast")
                for face in component:
                    processor.flip_face_normal(face)

    output_file = './Output/output_' + filename + '.obj'
    processor.write_obj_file(output_file)
    print(f"Mesh saved to {output_file}")

if __name__ == "__main__":
    main()
