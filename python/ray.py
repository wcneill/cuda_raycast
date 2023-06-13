import torch
import raycast

vertices = torch.tensor([[-1, 1, 0], 
                         [1, 1, 0], 
                         [-1, -1, 0.]], dtype=torch.float32).contiguous()

faces = torch.tensor([[2, 1, 0]]).int()

ray_origins = torch.tensor([[-0.5, 0.5, 2.2]]).contiguous()
ray_directions = -torch.eye(3, dtype=torch.float32)[:, -1].repeat(1, 1).contiguous()

# print(dir(raycast))
distances = raycast.find_distance(vertices.cuda(), faces.cuda(), ray_origins.cuda(), ray_directions.cuda())
intersects = raycast.find_intersections(vertices.cuda(), faces.cuda(), ray_origins.cuda(), ray_directions.cuda())

print(distances)
print(intersects)