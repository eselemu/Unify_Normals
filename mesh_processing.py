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
        self.meshComponents = []

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

    def get_connected_components(self):
        """
        Find connected components (submeshes) in the mesh.
        Each component is a set of connected Face objects.
        """
        visited = set()
        components = []
    
        for face in self.facesArray.values():
            if face in visited:
                continue
    
            # Start a new component
            component = set()
            stack = [face]
    
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
    
                # Traverse all neighboring faces via half-edge twins
                he = current.halfEdge
                start = he
                while True:
                    twin = he.twin
                    if twin and twin.face and twin.face not in visited:
                        stack.append(twin.face)
                    he = he.next
                    if he == start:
                        break
            components.append(component)
        self.meshComponents = components
        return components


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

    def correct_normals_for_component(self, component_faces, outwards=True):
        """Check and flip normals for a single component if needed."""
        # Recalculate vertex normals for this component
        # First, find highest vertex in component
        self.calculate_face_normals()
        self.calculate_vertex_normals()
        highest_vertex = None
        max_y = -float('inf')
        for face in component_faces:
            for vertex in face.vertices:
                if vertex.y > max_y:
                    max_y = vertex.y
                    highest_vertex = vertex
        
        if not highest_vertex or highest_vertex.normal is None:
            return  # Can't correct if data is missing
        
        # Calculate test point
        distance = 0.1
        test_point = np.array([
            highest_vertex.x + highest_vertex.normal[0] * distance,
            highest_vertex.y + highest_vertex.normal[1] * distance,
            highest_vertex.z + highest_vertex.normal[2] * distance
        ])
        
        is_outwards = test_point[1] > highest_vertex.y
        
        if outwards and not is_outwards:
            print("Flipping component to outward orientation")
            for face in component_faces:
                self.flip_face_normal(face)
            #self.flip_all_normals()
        elif not outwards and is_outwards:
            print("Flipping component to inward orientation")
            for face in component_faces:
                self.flip_face_normal(face)
            #self.flip_all_normals()
        
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
        """Flip the normal of a face and reverse its winding (half-edge order)"""
        if face.normal is not None:
            # Flip the normal vector
            face.normal = -face.normal
    
        # Collect half-edges in order
        he_list = []
        he = face.halfEdge
        start = he
        while True:
            he_list.append(he)
            he = he.next
            if he == start:
                break
    
        # Reverse the order of half-edges and update pointers
        n = len(he_list)
        for i in range(n):
            he_list[i].next = he_list[i - 1]
            he_list[i].prev = he_list[(i + 1) % n]
    
        # Update face.halfEdge to new "first" half-edge in reversed order
        face.halfEdge = he_list[-1]


    def unify_normals(self):
        """Unify face normals by propagating consistent orientation using DFS traversal."""
        visited = set()

        print("UNIFYING NORMALS")
        
        def dfs(face, expected_normal):
            visited.add(face)
            # If dot product is negative, flip face
            if np.dot(face.normal, expected_normal) < -0.25:
                print(face.normal)
                self.flip_face_normal(face)
                print(face.normal)
                print("----------------------------------------------------------------------------------------")
            
            # Traverse neighbor faces
            start_he = face.halfEdge
            he = start_he
            while True:
                twin = he.twin
                if twin and twin.face and twin.face not in visited:
                    #print("Entering recursive dfs")
                    dfs(twin.face, face.normal)
                he = he.next
                if he == start_he:
                    break

        # Start DFS from each unvisited face (handles disconnected components)
        for face in self.facesArray.values():
            if face not in visited:
                dfs(face, face.normal)

    def unify_normals_component(self, component_faces):
        """Unify normals in a subset of faces (single connected component)."""
        visited = set()
    
        def dfs(face, expected_normal):
            visited.add(face)
            if np.dot(face.normal, expected_normal) < -0.25:
                self.flip_face_normal(face)
    
            he = face.halfEdge
            start = he
            while True:
                twin = he.twin
                if twin and twin.face in component_faces and twin.face not in visited:
                    dfs(twin.face, face.normal)
                he = he.next
                if he == start:
                    break
        for face in component_faces:
            if face not in visited:
                dfs(face, face.normal)


    def unify_normals_all_components(self):
        """Detect all components and unify normals per component."""
        components = self.get_connected_components()
        print(f"Found {len(components)} connected components")

        for idx, component in enumerate(components):
            print(f"Unifying normals in component {idx}")
            self.unify_normals_component(component)


    def ray_intersects_triangle(self, origin, direction, v0, v1, v2, epsilon=1e-6):
        """
        Möller–Trumbore intersection algorithm.
        Returns True if ray intersects triangle (v0, v1, v2)
        """
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(direction, edge2)
        a = np.dot(edge1, h)
        if -epsilon < a < epsilon:
            return False  # Ray is parallel to triangle
    
        f = 1.0 / a
        s = origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False
    
        q = np.cross(s, edge1)
        v = f * np.dot(direction, q)
        if v < 0.0 or u + v > 1.0:
            return False
    
        t = f * np.dot(edge2, q)
        return t > epsilon  # Intersection exists


    def evaluate_orientation_by_raycast(self, component_faces, other_faces=None, num_samples=20, normal_scale=0.1):
        """
        Evaluates which orientation of normals leads to fewer collisions (self and optional other components).
        If other_faces is None, self-collisions are used for comparison only.
        """
    
        def count_intersections(faces_to_test, flip_normals=False):
            hits = 0
            for face in faces_to_test:
                verts = self.get_face_vertices(face)
                if len(verts) != 3:
                    continue  # Skip non-triangle faces
    
                v0, v1, v2 = [np.array([v.x, v.y, v.z]) for v in verts]
                face_normal = face.normal
                if flip_normals:
                    face_normal = -face_normal
    
                center = (v0 + v1 + v2) / 3
                ray_origin = center + face_normal * 0.01  # Offset to avoid self-hit
                ray_dir = face_normal
    
                # Test against all triangles (except self-face)
                for other in all_faces:
                    if other == face:
                        continue
                    ov0, ov1, ov2 = [np.array([v.x, v.y, v.z]) for v in self.get_face_vertices(other)]
                    if self.ray_intersects_triangle(ray_origin, ray_dir, ov0, ov1, ov2):
                        hits += 1
                        break  # Only count 1 hit per ray
            return hits
    
        all_faces = list(component_faces)
        if other_faces:
            all_faces += list(other_faces)
    
        # Count intersections for original and flipped normals
        hits_normal = count_intersections(component_faces, flip_normals=False)
        hits_flipped = count_intersections(component_faces, flip_normals=True)
    
        print(f"Component raycast collisions: normal={hits_normal}, flipped={hits_flipped}")
    
        return (hits_normal > 0 or hits_flipped > 0), hits_flipped - hits_normal  # True → keep current orientation (outwards), False → flip it


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
        # Connect twins (handles flipped faces)
        edge_map = {}

        for face_index, face_vertices in enumerate(self.faces):
            for i in range(3):
                start_idx = face_vertices[i]
                end_idx = face_vertices[(i + 1) % 3]
                key = tuple(sorted((start_idx, end_idx)))  # unordered edge key
                halfEdge = self.halfEdgesArray[face_index * 3 + i]

                if key in edge_map:
                    twin = edge_map[key]
                    halfEdge.twin = twin
                    twin.twin = halfEdge
                else:
                    edge_map[key] = halfEdge

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
