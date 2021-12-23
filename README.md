# mj117-nkatha-ujmusku-a3
a3 created for mj117-nkatha-ujmusku

# Part 1: Part-of-speech tagging

### Problem Statement : 
To correctly identify the tag for the word in any given sentence. 

### Approach :
Initially we need to get the probabilities for words, tags, words|tags (which is emission probability) and tags|tags (which is transition probability). SO when we get the train data, we run the train method in pos_solver file to lean and store all the probabilities. These values are stored in a universal dictionaries in the pos_solver files. Next, we calculate the probabilities for the test data sentences with the below methods:
  1. Simplified:
      1. In simplified, We first defined a list with all nouns as the values. 
      2. Then to calculate P(si|wi), we just multiply the P(wi|si) and P(si) and update the list.
      3. After we have iterated through the whole sentence and updated the list, we return the same list.
  2. HMM (Viterbi):
      1. In Viterbi, we first create a 12 x (len(sentence)) matrix. 
      2. For the first column, we calculate the probability for P(si|wi) based on the emission probabilities * porbability of the tag.
      3. For the next column onwards till the last column, we calculate the P(si|wi) row-wise based on the emission probability * transition probability * previous column values. 
      4. We store the max value that we get for every row in that column and its corrsponding tag in V_table and which_table.
      5. After that, we start to back track the values from the last column by checking the maximum in the column and then store the corresponding tag in another list. 
      6. And later based on that, we get the values from the other columns and store in the list. In the end, we return the list with the final tags. 
  3. Complex MCMC :
      1. For MCMC, we first generate 800 samples of the sentence and store it in the samples list.
      2. In these samples, we calculate the probalities based on the following conditions :
          1. if index = 0, then the probability only depends on the emission probability and probaility of the tag.
          2. if index = 1, then the porbability is calculated based on the emission probability and the transition probability from the earlier tag to current tag
          3. for rest all the values, the probability is calculated based on the emission probability, transition probability from the earlier tag to current tag and transition probability from the tag before earlier tag to current tag. 
          4. Also we calculate this for all the tags for a particular word and store the tag value in the sample list based on the max Value of the probabilities. 
      3. After calculating everything and storing it in the samples list, we then count the (wi|si) by iterating over the samples list starting from 300th sample onwards. The values before it are considered as the burn-in samples which we use for training the method. The count values are stored in a dictionary of dictionary which has the main key as word and the tags as the keys of the inner dictionary.
      4. Later, we just take the max of the value from the counts, and return that tag into the final list. 

### Difficulties faced :
1. The main issue I faced was in the complex mcmc method. With the complicated network that we had been given, it was a bit confusing to figure which transition probabilites had to be considered. 
2. After, trying different combinations and checking for the accuracy alongwith it, I found that the best accuracy was achieved when we just stuck to the arrows that were incoming for the particular tag and the emission probability for (wi|si).

### Conclusion : 
HMM (Viterbi) works the best out of all the three methods which gives the best sentence accuracy and word accuracy which is at par with the simplified version. Due to lack of time constraint, we had to reduce the number of samples for Complex MCMC, otherwise the results of Complex MCMC would increase with more number of samples and be at par with the Viterbi results. The final results that I get on the bc.test file is :

                  Simple     HMM Complex it's  late  and   you   said  they'd be    here  by    dawn  ''    .    
0. Ground truth   -96.12 -2189.26 -1728.71 prt   adv   conj  pron  verb  prt    verb  adv   adp   noun  .     .
1. Simple         -85.16 -2206.46 -1743.13 prt   adj   conj  pron  verb  prt    verb  adv   adp   noun  .     .
2. HMM            -85.16 -2206.46 -1743.13 prt   adj   conj  pron  verb  prt    verb  adv   adp   noun  .     .
3. Complex        -85.16 -2206.46 -1743.13 prt   adj   conj  pron  verb  prt    verb  adv   adp   noun  .     .    

==> So far scored 2000 sentences with 29442 words.


                   Words correct:     Sentences correct: 
                   
   0. Ground truth:      100.00%              100.00%
   1. Simple:             92.72%               42.05%
   2. HMM:                92.70%               47.20%
   3. Complex:            91.75%               39.95%

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
  


### PART 3 : Reading text:

### Problem Statement :

Recognizing the Image text and display them as plain text .

### Approach:

# 1.Simple Bayes Net :

1.	Similar  to the DAG graph , every part of the test image is divided based on the pixel dimensions 16*25 
2.	Each Training image with label  is converted to the to * and space and is compared with the test image pixel dimension . 
3.	Based on this,  probability is determined for each test cell . 
4.	In the Simple Bayes Net , as the variables weight is not dependent on each other . The above probability calculations was needed to predict the test words . 

# 2. HMM - VITERBI:
1. Coming to the Hidden Markov model Viterbi , the initial probabilities, transition and emission probability are calculated . 
2. Here the Transition probability is calculated for each current state variable with the all varaiables.
3. As observed the Simple bayes performed well for every test image . I have considered TOP 4 probable characters based on the initial probabilities . 
4. For each probable character i.e emission maintain them in the matrix and add the transition probabilities and added weight to these probabilities . 
5. Traversing through the matrix to predict the character of the test image . 

# Difficulties  Faced :
1.	What weights to be added to improve the performance of the model .
2.	How to handle spaces 
3.	The measures to take when the probability of the variable becomes zero . 
