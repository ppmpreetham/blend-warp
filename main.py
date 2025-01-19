import bpy
import warp as wp
import warp.sim
import numpy as np

class GranularSimulation:
    def __init__(self, num_frames=100, fps=60):
        self.num_frames = num_frames
        self.fps = fps
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.radius = 0.1

        # Model: Model Builder
        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius

        # Particles go here
        builder.add_particle_grid(
            dim_x=10,
            dim_y=10,
            dim_z=10,
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            pos=wp.vec3(0.0, 2.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, -1.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.integrator = wp.sim.SemiImplicitIntegrator()

    def simulate_frame(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt
        return self.state_0.particle_q.numpy()

def visualize_in_blender(simulation):
    collection_name = "GranularSimulation"
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)

    for frame in range(simulation.num_frames):
        bpy.context.scene.frame_set(frame + 1)

        positions = simulation.simulate_frame() # particle position calculations

        # Updation of particles in Blender
        for idx, pos in enumerate(positions):
            obj_name = f"Particle_{idx}"
            if obj_name in bpy.data.objects:
                obj = bpy.data.objects[obj_name]
            else:
                mesh = bpy.data.meshes.new(obj_name)
                obj = bpy.data.objects.new(obj_name, mesh)
                collection.objects.link(obj)

            obj.location = (pos[0], pos[1], pos[2])
        print(f"Frame {frame + 1}/{simulation.num_frames} complete.")

if __name__ == "__main__":
    sim = GranularSimulation(num_frames=100, fps=60)
    visualize_in_blender(sim)
