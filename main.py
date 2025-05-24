from mesh_processing import MeshProcessor

def main():
    # Initialize mesh processor
    processor = MeshProcessor()
    # Read input file
    input_file = './Input/s01.obj'
    processor.read_obj_file(input_file)
    
    # Convert to HEDS
    processor.convert_to_heds()
    
    # Calculate face normals
    processor.calculate_face_normals()
    
    # Color faces by orientation
    processor.color_faces_by_orientation()
    
    # Visualize with colored faces and normals
    processor.visualize_mesh_with_normals(show_normals=True)

    # Unify normals
    processor.unify_normals()

    # Color faces by orientation
    processor.color_faces_by_orientation()
    
    # Visualize with colored faces and normals
    processor.visualize_mesh_with_normals(show_normals=True)

    processor.correct_normals()

    processor.visualize_mesh_with_normals(show_normals=True)
    
    # Write output with colors (OBJ + MTL)
    '''output_file = 'output_colored.obj'
    processor.write_obj_with_mtl(output_file)
    print(f"Colored mesh saved to {output_file} and associated .mtl file")'''

if __name__ == "__main__":
    main()
