# Part 2: Ice tracking

### Problem statement :
Identify the ice layers(air-ice boundary and ice-rock boundary) in an image.
### Approach :
1. The image is converted to 175x225 matrix with pixels values in each row and column.
2. The problem is solved using three different approaches. 
  #### Simple Bayes net: 
  1. Following the figure (1.b) as bayes net, we just took the maximum of the edge strength in each column to identify the potential air ice boundary. Edge strength value helps us determine the sharp drop in pixel values.
  2. Similarly, for the ice-rock boundary, we assigned all the row values for each column above the air-ice boundary row(and +10) of that particular column to "-1".
  3. Then, we take the maximum of the edge strength of the edited matrix to find the next drop in edge strength from a+10th column till 175th row for each column.
  Following are the results of each test images:
  
  AirIce Boundary (image 30 - image 9 - image 16 - image 31 - image 23) - Simple
  ![image](https://media.github.iu.edu/user/18351/files/ae1f5780-52c4-11ec-8bd2-58bfa69d0161)

  Icerock Boundary (image 30 - image 9 - image 16 - image 31 - image 23) - Simple
  ![image](https://media.github.iu.edu/user/18351/files/b8d9ec80-52c4-11ec-8486-c77b16c4606e)

  We can clearly observe that there is a scope of improvement to capture the boundaries in an even better precision.
  
  #### HMM (Viterbi):
  1. Following the figure (1.a), we run the viterbi algorithm to obtain the most probable boundary in each column.
  2. We need three different probabilities to build Viterbi table - Emission probability, Transition probability, and initial probability. 
  3. Emission probability is the same as the one we used in simple bayes net. One addition to this is that I normalized the values in each column using min max scaling to scale the existing emission probabilities to a value between 0 and 1.
  4. Initial Probability is going to 0.8 for the row where the emission probability of the first column is highest. For all the other rows, it is strictly 0.2
  5. For Transition probability, I used the gaussian distribution function to assign maximum value for the row x in the current column when a boundary is observed in the row x in the previous column.
  6. For the rows x-1 and x+1, we have the slightly lower values than row x. Similarly, the rows x-2 and x+2, we obtain even lower values than x-1 and x+1. The values basically follow normal distribution with a mean(peak) as the previous boundary row.
  7. For airice boundary,using the above probabilities, we fill the viterbi table and then backtrack from the last column to come up with row values(/boundary).
  8. For icerock boundary, we use the same logic used in Simple Bayes net to convert all the rows of each column above the air-ice boundary(and +10) to 0. On this data, we run the viterbi algorithm to calculate the viterbi table and then backtracking.
 Following are the results of each test images:
  
  AirIce Boundary (image 30 - image 9 - image 16 - image 31 - image 23) - Simple(yellow) + HMM(blue)
  ![image](https://media.github.iu.edu/user/18351/files/b75df380-52c7-11ec-8f48-d95326fb382e)

  Icerock Boundary (image 30 - image 9 - image 16 - image 31 - image 23) - Simple(yellow) + HMM(blue)
  ![image](https://media.github.iu.edu/user/18351/files/c775d300-52c7-11ec-902f-f3765097a378)
  
  We observe more precised boundaries compared to Simple Bayes Net. Please observe image 9 to compare Simple with Viterbi. However, for image 23, we dont see a good precisioned ice-rock boundary.
  
  #### Human Feedback (Viterbi with inputs)
  1. We use the exact same procedure as above. However, there are two changes.
  2. Since we know the human feedback point on the boundary, the first change is that we run Viterbi algorithm from the human feedback column till end of the image(last column). Since we are 100% sure that human feedback point is true, we take the initial probability as 1 for the specific row and 0 for all the other rows in that column.
  3. The second change is that, we flip the matrix from the human feedback point to the start of the image(0th column). We run viterbi on the flipped matrix with initial probability of the human feedback point as 1 for that row.
  4. After running Viterbi on these two parts seperately, we flip the result of the second change back to normal and combine this with the first change output. 
  Following are the results of each test images:
  
  AirIce Boundary (image 30 - image 9 - image 16 - image 31 - image 23) - Simple(yellow) + HMM(blue) + Human Feedback(Red) 
  ![image](https://media.github.iu.edu/user/18351/files/e9c51c00-52da-11ec-997a-922684065867)
  
  ![image](https://media.github.iu.edu/user/18351/files/6eb03580-52db-11ec-973a-08dbc9c9add2)

  Icerock Boundary (image 30 - image 9 - image 16 - image 31 - image 23) - Simple(yellow) + HMM(blue) + Human Feedback(Red)
  ![image](https://media.github.iu.edu/user/18351/files/8f2ac080-52d8-11ec-8239-dcd6e63f3c7b)

  ![image](https://media.github.iu.edu/user/18351/files/838aca00-52d7-11ec-8bf1-60f520453f43)
  
  We observe precised boundaries compared to HMM and Simple. Please observe image 23 to compare Viterbi with human feedback.
  
  ### Difficulties faced:
  1. I spent quite some time to figure out the row and column human feedback input. The code throws an error while plotting the human feedback point if we give rows first followed by column. This is because x-axis is essentially is the column number and y-axis is the row number in the human feedback input. I figured this out after plotting the points. Changed the code accordingly.
  2. To come up with the transitional probabililties, I initially tried with fixed set of probabilities (0.9, 0.4, 0.03 etc), but later, after brainstorming with peers, I figured out gaussian distribution would give even better distribution than fixed value.
  3. For image 23, the ice-rock boundary was not very precise. I restricted the transitional probabilities even further by reducing the farther rows by 30%. Emission probabilities of the farther rows are further reduced by 5%. After doing these changes, I could see better results compared to earlier.
