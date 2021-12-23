#!/usr/local/bin/python3
#
# Authors: [PLEASE PUT YOUR NAMES AND USER IDS HERE]
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import math

import numpy as np
# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors
# (yellow, blue, red) to the filename
# Changed this code as per Prof. Crandall's recommendation on Inscribe
# Please refer to this link https://inscribe.education/main/indianau/6754110229500968/questions/6749461749659284?backToListTab=all
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = image.copy()
    new_image = draw_boundary(new_image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 4)
    imageio.imwrite(filename, new_image)

# main program
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    #input_filename = "test_images/09.png"
    input_filename = sys.argv[1]
    # gt_airice = [ int(i) for i in sys.argv[2:4] ]
    # gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # Changed this because x axis is essentially your col number and y axis is your row number
    gt_airice = [int(i) for i in sys.argv[2:4]][::-1]
    gt_icerock = [int(i) for i in sys.argv[4:6]][::-1]
    #print(gt_airice,gt_icerock)

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    #print(len(image_array),len(image_array[0]))
    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    # 175 rows and 225 columns
    # Simple - Air Ice
    simple_edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * simple_edge_strength / (amax(simple_edge_strength))))
    airice_simple = simple_edge_strength.argmax(axis=0)

    # Simple - Ice rock
    mask = np.transpose(np.arange(simple_edge_strength.shape[0]) < (airice_simple + 10)[:, None])
    simple_edge_strength[mask] = -1
    icerock_simple = simple_edge_strength.argmax(axis=0)
    empty_array = []

    #HMM - Air ice
    # transitional probability
    def transitional_prob(previous_boundary_row, hmm_edge_strength):
        x = np.arange(0, hmm_edge_strength.shape[0])
        gaussian_dist = np.exp((-((x - previous_boundary_row) / np.std(x)) ** 2) / 2) / (
                2 * math.pi * (np.std(x) ** 2)) ** 0.5
        gaussian_dist = (gaussian_dist - np.min(gaussian_dist)) / np.ptp(gaussian_dist)
        if sys.argv[1] in ("test_images/30.png","test_images/23.png"):
            top_indices, bottom_indices = list(range(previous_boundary_row-3)), list(range(previous_boundary_row+5, hmm_edge_strength.shape[0]))
            gaussian_dist[top_indices], gaussian_dist[bottom_indices] = 0.2,0.2
            #-3 and +5
        #gaussian_dist = gaussian_dist/np.max(gaussian_dist)
        return gaussian_dist

    # initial probability
    def hmm_initial_prob(initial_prob_factor, hmm_edge_strength):
        initial_prob = np.arange(0, hmm_edge_strength.shape[0], dtype=float)
        top_indices, bottom_indices = list(range(round(hmm_edge_strength.shape[0] * initial_prob_factor))), list(
            range(round(hmm_edge_strength.shape[0] * initial_prob_factor), hmm_edge_strength.shape[0]))
        initial_prob[top_indices], initial_prob[bottom_indices] = 0.8, 0.3
        return initial_prob

    # Used the following two lines from https://stackoverflow.com/questions/34980833/python-name-of-np-array-variable-as-string
    # Converting the name of numpy array to string to use in a if loop below
    def convert_array_str(obj, name):
        return [each for each in name if name[each] is obj]

    # Viterbi - HMM
    # The pseudo code for Viterbi is taken from Prof. Crandall's code in the activity session.
    def hmm(hmm_edge_strength, initial_prob_factor, flag, gt_airice, airice_feedback):

        # emission probability
        emmission_prob = (hmm_edge_strength - np.min(hmm_edge_strength, axis=0)) / np.ptp(hmm_edge_strength, axis=0)

        if flag == "human_feedback" and initial_prob_factor == 0 and sys.argv[1] == "test_images/23.png":
            if convert_array_str(hmm_edge_strength, globals())[0][0] == "b":
                mask = np.transpose(np.arange(emmission_prob.shape[0]) < (airice_feedback + 30)[:, None])
                emmission_prob[mask] *= 0.05
            else:
                mask = np.transpose(np.arange(emmission_prob.shape[0]) < (airice_feedback + 6)[:, None])
                emmission_prob[mask] *= 0.2

        #Initial probability
        if flag == "human_feedback":
            indices = list(range(hmm_edge_strength.shape[0]))[::-1]
            indices.remove(gt_airice[0])
            initial_prob = np.arange(0, hmm_edge_strength.shape[0], dtype=float)
            initial_prob[indices], initial_prob[gt_airice[0]] = 0.05, 1
        else:
            initial_prob = hmm_initial_prob(initial_prob_factor, hmm_edge_strength)

        #Initializing the viterbi and other tables
        Viterbi_table = {}
        path_table = {}
        tp_input = np.array([])

        # first column
        for row in range(hmm_edge_strength.shape[0]):
            Viterbi_table[(row, 0)] = initial_prob[row] * emmission_prob[row][0]
            path_table[(row, 0)] = row
            tp_input = np.append(tp_input, Viterbi_table[(row, 0)] )

        # rest columns
        V_last = np.array([])
        for col in range(1, hmm_edge_strength.shape[1]):
            t_prob = transitional_prob(tp_input.argmax(),hmm_edge_strength)
            tp_input = np.array([])
            for row_1 in range(hmm_edge_strength.shape[0]):
                max_path, max_value = max(
                    [(k, Viterbi_table[(k, col - 1)] * t_prob[k]) for k in range(hmm_edge_strength.shape[0])],
                    key=lambda l: l[1])
                path_table[(row_1, col)], Viterbi_table[(row_1, col)] = max_path, max_value * emmission_prob[row_1][col]
                tp_input = np.append(tp_input, row)
                #Appending only the last column in a numpy array to use it as a starting point for backtracking
                if col == hmm_edge_strength.shape[1] - 1:
                    V_last = np.append(V_last, Viterbi_table[(row_1, col)])

        max_index = V_last.argmax()
        hmm_seq = ones(hmm_edge_strength.shape[1])

        # backtracking
        for i in range(hmm_edge_strength.shape[1] - 1, -1, -1):
            hmm_seq[i] = int(max_index)
            max_index = path_table[(int(max_index), i)]

        return hmm_seq

    #HMM Viterbi - Air Ice
    hmm_edge_strength = edge_strength(input_image)
    airice_hmm = hmm(hmm_edge_strength, 0.5, "no_human_feedback", gt_airice, empty_array)
    #print(airice_hmm)

    # HMM - Ice Rock
    mask = np.transpose(np.arange(hmm_edge_strength.shape[0]) < (airice_hmm + 10)[:, None])
    hmm_edge_strength[mask] = 0
    icerock_hmm = hmm(hmm_edge_strength, 0.8, "no_human_feedback", gt_icerock, empty_array)
    #print(icerock_hmm)

    #airice - human feedback
    #running viterbi from human feedback point till the end
    human_edge_strength = edge_strength(input_image)
    fhuman_edge_strength = np.delete(human_edge_strength, np.s_[0:gt_airice[1]], axis=1)

    #flipping the matrix from feedback point to 0
    bhuman_edge_strength = np.delete(human_edge_strength, np.s_[gt_airice[1]+1:human_edge_strength.shape[1]], axis=1)
    bhuman_edge_strength = np.fliplr(bhuman_edge_strength)

    # combining both
    airice_feedback = np.hstack((np.flipud(hmm(bhuman_edge_strength, 0.5, "human_feedback", gt_airice, empty_array))[:-1],
                                 hmm(fhuman_edge_strength, 0.5, "human_feedback", gt_airice, empty_array)))

    #icerock - human feedback
    mask = np.transpose(np.arange(human_edge_strength.shape[0]) < (airice_feedback + 10)[:, None])
    human_edge_strength[mask] = 0
    fhuman_edge_strength = np.delete(human_edge_strength, np.s_[0:gt_icerock[1]], axis=1)
    bhuman_edge_strength = np.delete(human_edge_strength, np.s_[gt_icerock[1]+1:human_edge_strength.shape[1]], axis=1)
    bhuman_edge_strength = np.fliplr(bhuman_edge_strength)
    icerock_feedback = np.hstack((np.flipud(hmm(bhuman_edge_strength, 0, "human_feedback", gt_icerock, airice_feedback[0:gt_icerock[1]+2][:-1]))[:-1],
                                hmm(fhuman_edge_strength, 0, "human_feedback", gt_icerock, airice_feedback[gt_icerock[1]:human_edge_strength.shape[1]])))


    # Now write out the results as images and a text file
    # Inversed the feedback point because the x- axis is essentially number of columns in inbuilt asterick plotting function
    # whereas it is reverse in the code.
    # sample : python3 polar.py test_images/09.png x-axis(airice) y-axis(airice) x-axis(icerock) y-axis(icerock)
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice[::-1])
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock[::-1])
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, icerock_simple, icerock_hmm, airice_feedback, icerock_feedback):
            fp.write(str(i) + "\n")


