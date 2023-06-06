import torch
import raycast_distance

vertices = torch.tensor([[-1, 1, 0], 
                         [1, 1, 0], 
                         [-1, -1, 0.]], dtype=torch.float32).contiguous()

faces = torch.tensor([[2, 1, 0]]).int()

ray_origins = torch.tensor([[-0.5, 0.5, 1]]).contiguous()
ray_directions = -torch.eye(3, dtype=torch.float32)[:, -1].repeat(1, 1).contiguous()

answer = raycast_distance.measure_distance(vertices.cuda(), faces.cuda(), ray_origins.cuda(), ray_directions.cuda())
print(answer)