import tensorflow as tf
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import sys
import random

def init_population_from_file(filename):
    file = open(filename, 'r')
    population = np.loadtxt(filename)
    return population

def init_random_population(width=None, height=None, size=None):
    if width is None:
        width = random.randint(20,30)
    if height is None:
        height = random.randint(20,30)
    if size is None:
        size = random.randint(5, 20)
    population = np.zeros([height, width])
    for i in range(0, size):
        population[random.randint(0, height - 1)][random.randint(0, width - 1)] = 1
    return population

def make_frame(frame_id, plot, sess, population, new_population, height, width):
    sess.run(new_population)
    new_popul_result = sess.run(tf.reshape(new_population, [height, width]))
    sess.run(tf.assign(population, new_population))
    plot.set_array(new_popul_result)
    return plot

def main(argv):
    population = \
        init_random_population(sys.argv[1], sys.argv[2], sys.argv[3]) if (len(sys.argv) == 4) else \
        init_population_from_file(sys.argv[1]) if (len(sys.argv) == 2) else \
        init_random_population()
    height = population.shape[0]
    width = population.shape[1]
    population_initial = tf.constant(population, name='InitialPopulation')
    population = tf.reshape(population_initial, [1, height, width, 1], name='InitPopReshape')
    population = tf.cast(population, tf.float32, name='InitPopCast')
    population = tf.Variable(population, name='VarPopulation')
    kernel = tf.ones([3,3], name='Kernel')
    kernel = tf.reshape(kernel, [3,3,1,1], name='KernReshape')
    conv = tf.nn.conv2d(population, kernel, [1, 1, 1, 1], "SAME", name='Conv2d')
    neighbours = tf.subtract(conv, population, name='Nb_Sub')
    one = tf.constant(1, tf.float32, name='One')
    two = tf.constant(2, tf.float32, name='Two')
    three = tf.constant(3, tf.float32, name='Three')
    was_alive = tf.equal(population, one, 'EqWasAlive')
    have_2_nb = tf.equal(neighbours, two, 'Eq2Nb')
    alive = tf.logical_and(was_alive, have_2_nb, name='And_Alive')
    appeared = tf.equal(neighbours, three, name='Eq_Appeared')
    new_population = tf.logical_or(alive, appeared, name='Or_NewPopulation')
    new_population = tf.cast(new_population, tf.float32, name='NewPopCast')
    init = tf.global_variables_initializer()
    fig = plt.figure()
    fig.suptitle(sys.argv[1] if len(sys.argv) == 2 else 'Random')
    with tf.Session() as sess:
        sess.run(init)
        sess.run(new_population)
        new_popul_result = sess.run(tf.reshape(new_population, [height, width]))
        sess.run(tf.assign(population, new_population))
        plot = plt.imshow(new_popul_result, cmap='Greys', interpolation='nearest')
        anim = animation.FuncAnimation(fig, make_frame, fargs=(plot, sess, population, new_population, height, width), interval=100)
        plt.show()
        tf.summary.FileWriter("./logs", sess.graph)

if __name__ == "__main__":
    main(sys.argv)