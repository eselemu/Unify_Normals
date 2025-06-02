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
        """Vertex class storing position, normal and half-edge reference"""
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.halfEdge = None
            self.normal = None

    class Face:
        """Face class storing normal, color and half-edge reference"""
        def __init__(self):
            self.halfEdge = None
            self.color = (0.5, 0.5, 0.5, 0.5)
            self.vertices = []
            self.normal = None
    
    class HalfEdge:
        """Half-edge data structure for mesh representation"""
        def __init__(self):
            self.origin = None
            self.twin = None
            self.face = None
            self.next = None
            self.prev = None

    def calculate_face_normals(self):
        """Calculate normals for all faces using cross product of edges"""
        for face in self.facesArray.values():
            v0 = face.halfEdge.origin
            v1 = face.halfEdge.next.origin
            v2 = face.halfEdge.next.next.origin
            
            vec1 = np.array([v1.x - v0.x, v1.y - v0.y, v1.z - v0.z])
            vec2 = np.array([v2.x - v0.x, v2.y - v0.y, v2.z - v0.z])
            
            normal = np.cross(vec1, vec2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            face.normal = normal

    def calculate_vertex_normals(self):
        """Calculate vertex normals by averaging adjacent face normals"""
        for vertex in self.verticesArray.values():
            vertex.normal = np.zeros(3)
        
        for face in self.facesArray.values():
            if face.normal is None:
                continue
            for vertex in face.vertices:
                vertex.normal += face.normal
        
        for vertex in self.verticesArray.values():
            norm = np.linalg.norm(vertex.normal)
            if norm > 0:
                vertex.normal /= norm

    def get_connected_components(self):
        """Find connected components (submeshes) using DFS traversal"""
        visited = set()
        components = []
    
        for face in self.facesArray.values():
            if face in visited:
                continue
    
            component = set()
            stack = [face]
    
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
    
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
        
        for vertex in self.verticesArray.values():
            if vertex.y > max_y:
                max_y = vertex.y
                highest_vertex = vertex
        
        return highest_vertex

    def is_mesh_outwards(self, distance=0.1):
        """Check if mesh normals point outwards by testing highest vertex"""
        highest_vertex = self.get_highest_vertex()
        if not highest_vertex or highest_vertex.normal is None:
            return True
        
        test_point = np.array([
            highest_vertex.x + highest_vertex.normal[0] * distance,
            highest_vertex.y + highest_vertex.normal[1] * distance,
            highest_vertex.z + highest_vertex.normal[2] * distance
        ])
        
        return test_point[1] > highest_vertex.y

    def correct_normals(self, outwards=True):
        """Flip all normals if they don't match desired orientation"""
        is_outwards = self.is_mesh_outwards()

        if outwards and (not is_outwards):
            self.flip_all_normals()
        elif (not outwards) and is_outwards:
            self.flip_all_normals()

    def flip_all_normals(self):
        """Flip all face normals and adjust half-edge structures"""
        for face in self.facesArray.values():
            self.flip_face_normal(face)

    def color_faces_by_orientation(self, reference_normal=None):
        """Color faces based on their normal orientation (green=correct, red=flipped)"""
        if reference_normal is None:
            first_face = next(iter(self.facesArray.values()))
            reference_normal = first_face.normal
        
        green = True
        for face in self.facesArray.values():
            if face.normal is None:
                continue
                
            dot_product = np.dot(face.normal, reference_normal)
            if dot_product < 0:
                green = not green
            face.color = (0, 1, 0, 0.5) if green else (1, 0, 0, 0.5)
            reference_normal = face.normal

    def flip_face_normal(self, face):
        """Flip the normal of a face and reverse its winding order"""
        if face.normal is not None:
            face.normal = -face.normal
    
        he_list = []
        he = face.halfEdge
        start = he
        while True:
            he_list.append(he)
            he = he.next
            if he == start:
                break
    
        n = len(he_list)
        for i in range(n):
            he_list[i].next = he_list[i - 1]
            he_list[i].prev = he_list[(i + 1) % n]
    
        face.halfEdge = he_list[-1]

    def unify_normals(self):
        """Unify face normals using DFS traversal for single-component meshes"""
        visited = set()
        
        def dfs(face, expected_normal):
            visited.add(face)
            if np.dot(face.normal, expected_normal) < -0.25:
                self.flip_face_normal(face)
            
            he = face.halfEdge
            start = he
            while True:
                twin = he.twin
                if twin and twin.face and twin.face not in visited:
                    dfs(twin.face, face.normal)
                he = he.next
                if he == start:
                    break

        for face in self.facesArray.values():
            if face not in visited:
                dfs(face, face.normal)

    def unify_normals_component(self, component_faces):
        """Unify normals in a subset of faces (single connected component)"""
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

    def ray_intersects_triangle(self, origin, direction, v0, v1, v2, epsilon=1e-6):
        """Möller-Trumbore ray-triangle intersection test"""
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(direction, edge2)
        a = np.dot(edge1, h)
        if -epsilon < a < epsilon:
            return False
    
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
        return t > epsilon

    def evaluate_orientation_by_raycast(self, component_faces, other_faces=None, num_samples=20, normal_scale=0.1):
        """Evaluate normal orientation by counting ray collisions"""
        def count_intersections(faces_to_test, flip_normals=False):
            hits = 0
            for face in faces_to_test:
                verts = self.get_face_vertices(face)
                if len(verts) != 3:
                    continue
    
                v0, v1, v2 = [np.array([v.x, v.y, v.z]) for v in verts]
                face_normal = -face.normal if flip_normals else face.normal
    
                center = (v0 + v1 + v2) / 3
                ray_origin = center + face_normal * 0.01
                ray_dir = face_normal
    
                for other in all_faces:
                    if other == face:
                        continue
                    ov0, ov1, ov2 = [np.array([v.x, v.y, v.z]) for v in self.get_face_vertices(other)]
                    if self.ray_intersects_triangle(ray_origin, ray_dir, ov0, ov1, ov2):
                        hits += 1
                        break
            return hits
    
        all_faces = list(component_faces)
        if other_faces:
            all_faces += list(other_faces)
    
        hits_normal = count_intersections(component_faces, flip_normals=False)
        hits_flipped = count_intersections(component_faces, flip_normals=True)
    
        print(f"Component raycast collisions: normal={hits_normal}, flipped={hits_flipped}")
        return (hits_normal > 0 or hits_flipped > 0), hits_flipped - hits_normal

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

    def convert_to_heds(self):
        """Convert vertex-face representation to Half-Edge Data Structure"""
        self.verticesArray = {}
        for i, (x, y, z) in enumerate(self.vertices):
            self.verticesArray[i] = self.Vertex(x, y, z)

        self.halfEdgesArray = [self.HalfEdge() for _ in range(len(self.faces)*3)]
        self.facesArray = {}
        for i in range(len(self.faces)):
            self.facesArray[i] = self.Face()

        for face_index, face_vertices in enumerate(self.faces): 
            face = self.facesArray[face_index]
            for i, vertex_index in enumerate(face_vertices):
                vertex = self.verticesArray[vertex_index]
                face.vertices.append(vertex)
                halfEdge = self.halfEdgesArray[face_index * 3 + i]
                halfEdge.origin = vertex
                vertex.halfEdge = halfEdge

        for face_index in range(len(self.faces)):
            face = self.facesArray[face_index]
            face_halfEdgesArray = [self.halfEdgesArray[face_index * 3 + i] for i in range(3)]
            face.halfEdge = face_halfEdgesArray[0]
            for i in range(3):
                face_halfEdgesArray[i].face = face
                face_halfEdgesArray[i].next = face_halfEdgesArray[(i + 1) % 3]
                face_halfEdgesArray[i].prev = face_halfEdgesArray[(i + 2) % 3]

        edge_map = {}
        for face_index, face_vertices in enumerate(self.faces):
            for i in range(3):
                start_idx = face_vertices[i]
                end_idx = face_vertices[(i + 1) % 3]
                key = tuple(sorted((start_idx, end_idx)))
                halfEdge = self.halfEdgesArray[face_index * 3 + i]

                if key in edge_map:
                    twin = edge_map[key]
                    halfEdge.twin = twin
                    twin.twin = halfEdge
                else:
                    edge_map[key] = halfEdge

    def get_face_vertices(self, face):
        """Get vertices for a given face"""
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
        """Convert HEDS back to vertex-face representation"""
        vertices = [(vertex.x, vertex.y, vertex.z) for vertex in self.verticesArray.values()]
        faces = []
        for face in self.facesArray.values():
            face_vertices = self.get_face_vertices(face)
            face_indices = [list(self.verticesArray.keys())[list(self.verticesArray.values()).index(vertex)] 
                          for vertex in face_vertices]
            faces.append(face_indices)
        return vertices, faces

    def write_obj_file(self, output_file):
        """Write mesh to OBJ file"""
        vertices, faces = self.convert_to_vf()
        with open(output_file, 'w') as obj_file:
            for vertex in vertices:
                obj_file.write('v ' + ' '.join(map(str, vertex)) + '\n')
            for face in faces:
                obj_file.write('f ' + ' '.join(map(lambda x: str(x + 1), face)) + '\n')

    def visualize_mesh_with_normals(self, show_normals=True, normal_scale=0.1):
        """Visualize mesh with colored faces and optional normals"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        if all(face.normal is None for face in self.facesArray.values()):
            self.calculate_face_normals()
            self.color_faces_by_orientation()
    
        for face in self.facesArray.values():
            face_vertices = self.get_face_vertices(face)
            verts = [[v.x, v.y, v.z] for v in face_vertices]
            
            poly = Poly3DCollection([verts], alpha=0.5)
            poly.set_facecolor(face.color[:3])
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)
    
            if show_normals and face.normal is not None:
                center = np.mean(verts, axis=0)
                normal_end = center + face.normal * normal_scale
                
                ax.quiver(center[0], center[1], center[2],
                          face.normal[0], face.normal[1], face.normal[2],
                          length=normal_scale, color='b', arrow_length_ratio=0.1)
    
        vertices_array = np.array([(v.x, v.y, v.z) for v in self.verticesArray.values()])
        ax.set_box_aspect([np.ptp(vertices_array[:,0]), 
                         np.ptp(vertices_array[:,1]), 
                         np.ptp(vertices_array[:,2])])
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Mesh with Face Orientation (Green=Correct, Red=Flipped)')
        plt.show()
