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

    class Face:
        def __init__(self):
            self.halfEdge = None
            self.color = (0.5, 0.5, 0.5, 0.5)
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

    def color_faces_by_orientation(self, reference_normal=None):
        """Color faces based on their normal orientation"""
        if reference_normal is None:
            # Use the first face's normal as reference if none provided
            first_face = next(iter(self.facesArray.values()))
            reference_normal = first_face.normal
        
        green = False
        for face in self.facesArray.values():
            if face.normal is None:
                continue
                
            dot_product = np.dot(face.normal, reference_normal)
            if dot_product < 0:
                green = ~green
            if green:
                # Facing same direction as reference (green)
                face.color = (0, 1, 0, 0.5)  # RGBA
            else:
                # Facing opposite direction (red)
                face.color = (1, 0, 0, 0.5)  # RGBA
            reference_normal = face.normal

    def color_faces_by_propagated_orientation(self):
        """Color faces using consistent orientation propagation"""
        # Reset all colors to undetermined
        for face in self.facesArray.values():
            face.color = (0.5, 0.5, 0.5, 0.5)
        
        # Use a queue for BFS traversal
        from collections import deque
        queue = deque()
        
        # Start with first face as reference
        first_face = next(iter(self.facesArray.values()))
        queue.append(first_face)
        first_face.color = (0, 1, 0, 0.5)  # Green = correct
        
        visited = set()
        
        while queue:
            current_face = queue.popleft()
            if current_face in visited:
                continue
                
            visited.add(current_face)
            
            # Get all adjacent faces through half-edges
            start_he = current_face.halfEdge
            he = start_he
            while True:
                # Get twin edge's face (adjacent face)
                twin = he.twin
                if twin is not None:
                    adjacent_face = twin.face
                    
                    if adjacent_face not in visited:
                        # Check orientation consistency
                        dot_product = np.dot(current_face.normal, adjacent_face.normal)
                        
                        if dot_product < 0:  # Normals point in opposite directions
                            # Flip the adjacent face's normal
                            adjacent_face.normal *= -1
                            adjacent_face.color = (1, 0, 0, 0.5)  # Red = flipped
                        else:
                            adjacent_face.color = (0, 1, 0, 0.5)  # Green = consistent
                            
                        queue.append(adjacent_face)
                
                he = he.next
                if he == start_he:
                    break

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
            for i, vertex_index in enumerate(face_vertices):
                halfEdge = self.halfEdgesArray[face_index * 3 + i]
                halfEdge.origin = self.verticesArray[vertex_index]
                self.verticesArray[vertex_index].halfEdge = halfEdge

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

    def unify_normals(self):
        """Unify normals to consistent orientation"""
        # This is where you'll implement your normal unification algorithm
        # For now it's a placeholder
        print("Normal unification not yet implemented")
        pass
