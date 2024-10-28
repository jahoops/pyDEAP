import pygame
import pymunk
import random
import debugpy
import os
from deap import base, creator, tools, algorithms

# Check if debugging is enabled
if os.getenv("DEBUG") == "1":
    # Set up debugpy for remote debugging
    debugpy.listen(("0.0.0.0", 5680))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached.")

# Initialize PyGame and PyMunk
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()

# Define the fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float), 8)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def create_ramp(space):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Poly(body, [(0, 300), (200, 350), (400, 375), (570, 370), (600, 400), (630, 380)])
    space.add(body, shape)
    return shape

def create_obstacles(space, positions):
    obstacles = []
    for i in range(4):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Circle(body, 15)  # Solid circle with radius 15
        body.position = (positions[i*2] * 600, positions[i*2 + 1] * 200 + 50)  # Varying positions
        space.add(body, shape)
        obstacles.append(shape)
    return obstacles

def eval_individual(individual):
    positions = individual
    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 10))
    shape = pymunk.Circle(body, 10)
    body.position = (300, 50)  # Drop from the top center
    space = pymunk.Space()
    space.gravity = (0, 900)
    space.add(body, shape)

    ramp_shape = create_ramp(space)
    obstacle_shapes = create_obstacles(space, positions)

    hit_obstacles = [False, False, False, False]

    # Run simulation and check if the ball hits all obstacles
    for _ in range(500):  # Limit the iterations to a safe number
        space.step(0.02)
        if body.position.y > 600:
            return (0,)  # Ball fell off the screen, fitness is zero
        if body.position.y >= 380 and body.position.x >= 600:
            end_velocity = body.velocity.length
            num_hit = sum(hit_obstacles)
            return (end_velocity * num_hit,) if num_hit > 0 else (0,)
        for i, obstacle_shape in enumerate(obstacle_shapes):
            if shape.shapes_collide(obstacle_shape).points:
                hit_obstacles[i] = True

    return (0,)  # If the loop completes without hitting all obstacles, fitness is zero

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.2)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def draw_ramp(ramp_shape):
    points = [(int(x), int(y)) for x, y in ramp_shape.get_vertices()]
    pygame.draw.polygon(screen, (0, 255, 0), points)

def draw_obstacles(obstacle_shapes):
    for obstacle_shape in obstacle_shapes:
        pygame.draw.circle(screen, (0, 0, 255), (int(obstacle_shape.body.position.x), int(obstacle_shape.body.position.y)), 15)

def visualize_best(most_fit):
    space = pymunk.Space()
    space.gravity = (0, 900)
    positions = most_fit
    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 10))
    shape = pymunk.Circle(body, 10)
    body.position = (300, 50)  # Drop from the top center
    space.add(body, shape)

    ramp_shape = create_ramp(space)
    obstacle_shapes = create_obstacles(space, positions)

    # Run simulation and visualize the best individual
    for _ in range(500):  # Limit the iterations to a safe number
        space.step(0.02)
        if body.position.y > 600 or (body.position.y >= 380 and body.position.x >= 600):
            break
        screen.fill((0, 0, 0))
        draw_ramp(ramp_shape)
        draw_obstacles(obstacle_shapes)
        pygame.draw.circle(screen, (255, 0, 0), (int(body.position.x), int(body.position.y)), 10)
        pygame.display.flip()
        clock.tick(60)

# Main evolution loop
pop = toolbox.population(n=500)  # Increase population to 500
for gen in range(100):  # Run for 100 generations
    # Evaluate fitness without rendering
    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        if fit is not None:
            ind.fitness.values = fit
    pop = toolbox.select(offspring, k=len(pop))
    most_fit = tools.selBest(pop, 1)[0]

    # Print and visualize the best individual
    print(f"Generation {gen}: Max Velocity = {most_fit.fitness.values[0]}")
    visualize_best(most_fit)
    clock.tick(10)

pygame.quit()
