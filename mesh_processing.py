import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class MeshProcessor:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.verticesArray = {}
        self.halfEdgesArray = []
        self.facesArray = {}

    class Vertex:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.halfEdge = None
            self.normal = None

    class Face:
        def __init__(self):
            self.halfEdge = None
            self.color = (0.5, 0.5, 0.5, 0.5)
            self.vertices = []
            self.normal = None

    class HalfEdge:
        def __init__(self):
            self.origin = None
            self.twin = None
            self.face = None
            self.next = None
            self.prev = None

    def calculate_face_normals(self):
        """Calculate normals for all faces"""
        for face in self.facesArray.values():
            # Get three vertices of the face
            v0 = face.halfEdge.origin
            v1 = face.halfEdge.next.origin
            v2 = face.halfEdge.next.next.origin
            
            # Calculate vectors
            vec1 = np.array([v1.x - v0.x, v1.y - v0.y, v1.z - v0.z])
            vec2 = np.array([v2.x - v0.x, v2.y - v0.y, v2.z - v0.z])
            
            # Calculate cross product (normal)
            normal = np.cross(vec1, vec2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm  # Normalize
            face.normal = normal

    def calculate_vertex_normals(self):
        """Calculate vertex normals by averaging adjacent face normals"""
        # Reset all vertex normals
        for vertex in self.verticesArray.values():
            vertex.normal = np.zeros(3)
        
        # Sum all face normals for each vertex
        for face in self.facesArray.values():
            if face.normal is None:
                continue
                
            for vertex in face.vertices:  # Use stored vertices
                vertex.normal += face.normal
        
        # Normalize the vertex normals
        for vertex in self.verticesArray.values():
            norm = np.linalg.norm(vertex.normal)
            if norm > 0:
                vertex.normal /= norm
        
        return {v_idx: vertex.normal for v_idx, vertex in self.verticesArray.items()}

    def get_highest_vertex(self):
        """Find the vertex with the maximum Y-coordinate"""
        highest_vertex = None
        max_y = -float('inf')
        
        for v_idx, vertex in self.verticesArray.items():
            if vertex.y > max_y:
                max_y = vertex.y
                highest_vertex = vertex
        
        return highest_vertex

    def is_mesh_outwards(self, distance=0.1):
        # Get the highest vertex
        highest_vertex = self.get_highest_vertex()
        if not highest_vertex or highest_vertex.normal is None:
            return True  # No vertex or normal, assume correct
        
        # Calculate test point by moving along vertex normal
        test_point = np.array([
            highest_vertex.x + highest_vertex.normal[0] * distance,
            highest_vertex.y + highest_vertex.normal[1] * distance,
            highest_vertex.z + highest_vertex.normal[2] * distance
        ])
        
        # Compare Y components
        return test_point[1] > highest_vertex.y

    def correct_normals(self, outwards = True):
        # Check if normals need flipping
        is_outwards = self.is_mesh_outwards()

        if is_outwards:
            print("Mesh is outwards")
        else:
            print("Mesh is inwards")

        if outwards and (not is_outwards):
            print("Flipping all normals to ensure outward orientation")
            self.flip_all_normals()
        elif (not outwards) and is_outwards:
            print("Flipping all normals to ensure inward orientation")
            self.flip_all_normals()
    
    def flip_all_normals(self):
        """Flip all face normals and adjust half-edge structures"""
        print("Flipping all normals")
        for face in self.facesArray.values():
            self.flip_face_normal(face)

    def color_faces_by_orientation(self, reference_normal=None):
        """Color faces based on their normal orientation"""
        if reference_normal is None:
            # Use the first face's normal as reference if none provided
            first_face = next(iter(self.facesArray.values()))
            reference_normal = first_face.normal
        
        green = True
        for face in self.facesArray.values():
            if face.normal is None:
                continue
                
            dot_product = np.dot(face.normal, reference_normal)
            if dot_product < 0:
                green = not green
            if green:
                # Facing same direction as reference (green)
                face.color = (0, 1, 0, 0.5)  # RGBA
            else:
                # Facing opposite direction (red)
                face.color = (1, 0, 0, 0.5)  # RGBA
            reference_normal = face.normal

    def flip_face_normal(self, face):
        """Flip the normal of a face and adjust its half-edge structure"""
        if face.normal is not None:
            # Flip the normal vector
            face.normal = -face.normal
            
            # Get all half-edges of the face
            start_he = face.halfEdge
            half_edges = []
            he = start_he
            while True:
                half_edges.append(he)
                he = he.next
                if he == start_he:
                    break
            
            # Reverse the order of half-edges
            for i in range(len(half_edges)):
                # Update next and prev pointers
                half_edges[i].next = half_edges[(i - 1) % len(half_edges)]
                half_edges[i].prev = half_edges[(i + 1) % len(half_edges)]
            
            # Update face's half-edge pointer (optional, could keep same)
            face.halfEdge = half_edges[0]


    def unify_normals(self, reference_normal=None):
        """Color faces based on their normal orientation"""
        if reference_normal is None:
            # Use the first face's normal as reference if none provided
            first_face = next(iter(self.facesArray.values()))
            reference_normal = first_face.normal
        for face in self.facesArray.values():
            if face.normal is None:
                continue
                
            dot_product = np.dot(face.normal, reference_normal)
            if dot_product < 0:
                self.flip_face_normal(face)
            reference_normal = face.normal

    def read_obj_file(self, file_path):
        """Read vertices and faces from an OBJ file"""
        self.vertices = []
        self.faces = []
        
        with open(file_path, 'r') as obj_file:
            for line in obj_file:
                if line.startswith('v '):
                    vertex = list(map(float, line.strip().split()[1:]))
                    self.vertices.append(vertex)
                elif line.startswith('f '):
                    face = list(map(int, line.strip().split()[1:]))
                    face = [index - 1 for index in face]
                    self.faces.append(face)
        return self.vertices, self.faces

    def convert_to_heds(self):
        """Convert vertex-face representation to HEDS"""
        # Create vertices
        self.verticesArray = {}
        for i, (x, y, z) in enumerate(self.vertices):
            vertex = self.Vertex(x, y, z)
            self.verticesArray[i] = vertex

        # Create half edges
        self.halfEdgesArray = [self.HalfEdge() for _ in range(len(self.faces)*3)]

        # Create faces
        self.facesArray = {}
        for i, _ in enumerate(self.faces):
            face = self.Face()
            self.facesArray[i] = face

        # Connect halfEdges to vertices
        for face_index, face_vertices in enumerate(self.faces): 
            face = self.facesArray[face_index]
            for i, vertex_index in enumerate(face_vertices):
                # Store vertex reference in face
                vertex = self.verticesArray[vertex_index]
                face.vertices.append(vertex)
                # Connect half-edge
                halfEdge = self.halfEdgesArray[face_index * 3 + i]
                halfEdge.origin = vertex
                vertex.halfEdge = halfEdge

        # Connect half edges to faces
        for face_index, face_vertices in enumerate(self.faces):
            face = self.facesArray[face_index]
            face_halfEdgesArray = [self.halfEdgesArray[face_index * 3 + i] for i in range(3)]
            face.halfEdge = face_halfEdgesArray[0]
            for i in range(3):
                face_halfEdgesArray[i].face = face
                face_halfEdgesArray[i].next = face_halfEdgesArray[(i + 1) % 3]
                face_halfEdgesArray[i].prev = face_halfEdgesArray[(i + 2) % 3]

        # Connect twins
        edge_map = {}  # key: (start_vertex_index, end_vertex_index), value: halfEdge

        for face_index, face_vertices in enumerate(self.faces):
            for i in range(3):
                start_idx = face_vertices[i]
                end_idx = face_vertices[(i + 1) % 3]
                halfEdge = self.halfEdgesArray[face_index * 3 + i]
                edge_map[(start_idx, end_idx)] = halfEdge

        for (start, end), halfEdge in edge_map.items():
            twin_key = (end, start)
            if twin_key in edge_map:
                halfEdge.twin = edge_map[twin_key]

        return self.verticesArray, self.halfEdgesArray, self.facesArray

    def get_face_vertices(self, face):
        """Helper method to get vertices for a face"""
        vertices = []
        start_halfEdge = face.halfEdge
        current_halfEdge = start_halfEdge
        while True:
            vertices.append(current_halfEdge.origin)
            current_halfEdge = current_halfEdge.next
            if current_halfEdge == start_halfEdge:
                break
        return vertices

    def convert_to_vf(self):
        """Convert HEDS representation back to vertex-face"""
        vertices = [(vertex.x, vertex.y, vertex.z) for vertex in self.verticesArray.values()]
        faces = []
        for face in self.facesArray.values():
            face_vertices = self.get_face_vertices(face)
            face_indices = [list(self.verticesArray.keys())[list(self.verticesArray.values()).index(vertex)] 
                          for vertex in face_vertices]
            faces.append(face_indices)
        return vertices, faces

    def write_obj_file(self, output_file):
        """Write vertices and faces to an OBJ file"""
        vertices, faces = self.convert_to_vf()
        with open(output_file, 'w') as obj_file:
            for vertex in vertices:
                obj_file.write('v ' + ' '.join(map(str, vertex)) + '\n')
            for face in faces:
                obj_file.write('f ' + ' '.join(map(lambda x: str(x + 1), face)) + '\n')

    def visualize_mesh(self, vertex_color='k', edge_color='b'):
        """Visualize the mesh using matplotlib"""
        vertices, faces = self.convert_to_vf()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot vertices
        vertices_array = np.array(vertices)
        ax.scatter(vertices_array[:,0], vertices_array[:,2], vertices_array[:,1], 
                  c=vertex_color, depthshade=False)

        # Plot edges
        for face in faces:
            face_vertices = [vertices[i] for i in face]
            face_vertices.append(vertices[face[0]])  # Close the loop
            face_vertices = np.array(face_vertices)
            ax.plot(face_vertices[:,0], face_vertices[:,2], face_vertices[:,1], c=edge_color)  
            
        # Set equal aspect ratio
        ax.set_box_aspect([np.ptp(vertices_array[:,0]), 
                         np.ptp(vertices_array[:,0]), 
                         np.ptp(vertices_array[:,1])])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def visualize_mesh_with_normals(self, show_normals=True, normal_scale=0.1):
        """Visualize the mesh with colored faces and optional normals"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # Calculate face normals if not already done
        if all(face.normal is None for face in self.facesArray.values()):
            self.calculate_face_normals()
            self.color_faces_by_orientation()
    
        # Plot faces with colors
        for face in self.facesArray.values():
            face_vertices = self.get_face_vertices(face)
            verts = [[v.x, v.y, v.z] for v in face_vertices]
            
            # Create a polycollection for the face
            poly = Poly3DCollection([verts], alpha=0.5)
            poly.set_facecolor(face.color[:3])  # Use only RGB components
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)
    
            # Plot normals if requested
            if show_normals and face.normal is not None:
                # Calculate face center
                center = np.mean(verts, axis=0)
                normal_end = center + face.normal * normal_scale
                
                # Plot normal vector
                ax.quiver(center[0], center[1], center[2],
                          face.normal[0], face.normal[1], face.normal[2],
                          length=normal_scale, color='b', arrow_length_ratio=0.1)
    
        # Set equal aspect ratio
        vertices_array = np.array([(v.x, v.y, v.z) for v in self.verticesArray.values()])
        ax.set_box_aspect([np.ptp(vertices_array[:,0]), 
                         np.ptp(vertices_array[:,1]), 
                         np.ptp(vertices_array[:,2])])
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Mesh with Face Orientation (Green=Correct, Red=Flipped)')
        plt.show()

