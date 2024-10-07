## PCA (Principal Component Analysis, Dimensionality Reduction Process)

As the number of features or dimensions in a dataset increases, the amount of data required to obtain a statistically significant result increases exponentially. This can lead to issues such as overfitting, increased computation time, and reduced accuracy of machine learning models 📈. This is known as the curse of dimensionality problems that arise while working with high-dimensional data. 

### Story Time

🪄 Imagine your FM radio as a magical device. Inside, there's a world with many radio stations, each like a different door to music. Principal Component Analysis (PCA) is a bit like a wizard's spell for data, helping us make sense of complex information. In our radio world, it's like finding the most important doors while ignoring the ones we don't need. When you turn the radio knob, you're using a magical wand to pick the exact music door you want to open, leaving out the ones you don't want 🎵✨. And, just like making sure your favorite music is loud and clear, PCA and radio use tricks to make sure the music you love sounds great without noisy interruptions 🎶. So, while PCA doesn't really cast spells on radios, these ideas help us enjoy our radio adventures and understand our data better, making both more magical in our daily lives. 🧙‍♂️📻

### Overcoming Feature Overload 🚀

Is your dataset looking like a treasure chest with a mountain of features? 📦🏔️ Fear not! We have a solution to make your model training and testing faster while reducing the risk of overfitting. 🎯

### The Challenge 🤔

You're dealing with a dataset imported from sklearn library, and it's a behemoth with nearly 30 columns or features. 📈📉 But hold on! More features don't always mean better results. In fact, they can lead to slower model training and the dreaded overfitting monster. 😱

### The Solution 🛠️

Let's trim the fat and get lean and mean! We need a strategy to reduce the number of features. 

#### Feature Selection 🧐

Think of feature selection as panning for gold in a river of data. We'll sift through all those columns and only keep the nuggets that matter. 

## Step-By-Step Explanation of PCA (Principal Component Analysis) 🛤️

**Step 1: Standardization** 📏

First, we need to standardize our dataset to ensure that each variable has a mean of 0 and a standard deviation of 1.

`Z = {X - mu} / {sigma}`   

![Standardization](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-9a39cfcb7649d719ef13776cce3ecda0_l3.svg)

Here,

`mu` is the mean of independent features,  
`sigma` is the standard deviation of independent features. 

**Step 2: Covariance Matrix Computation** 📊

Covariance measures the strength of joint variability between two or more variables, indicating how much they change in relation to each other.

![Formula For Covariance Matrix Calculation](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a9f7569d00480192b75f3d38055e1529_l3.svg)

The value of covariance can be positive, negative, or zeros.

- Positive: As `x1` increases, `x2` also increases.
- Negative: As `x1` increases, `x2` also decreases.
- Zeros: No direct relation

**Step 3: Compute Eigenvalues and Eigenvectors of Covariance Matrix to Identify Principal Components** 🧮

Let `A` be a square `nXn` matrix and `X` be a non-zero vector for which 

![AX = lambda X](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-60384c5ce1abcb24694f89550aff6d9c_l3.svg)

for some scalar values `lambda`. Then `lambda` is known as the eigenvalue of matrix `A` and `X` is known as the eigenvector of matrix `A` for the corresponding eigenvalue.

It can also be written as :

![AX - lambda X = 0, (A - lambda I)X = 0](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-d484659a01a1e820f0b4c3cb1e9df756_l3.svg)

where `I` is the identity matrix of the same shape as matrix `A`. And the above conditions will be true only if `(A - lambda I)` will be non-invertible (i.e., a singular matrix). That means,

![|A - lambda I| = 0](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-be120def60ad50285342b93c3f1e5073_l3.svg)  

### Determine the Number of Principal Components 

We can either consider the number of principal components of any value of our choice or by limiting the explained variance. 

### Project the Data onto the Selected Principal Components

Find the projection matrix, It is a matrix of eigenvectors corresponding to the largest eigenvalues of the covariance matrix of the data. it projects the high-dimensional dataset onto a lower-dimensional subspace
The eigenvectors of the covariance matrix of the data are referred to as the principal axes of the data, and the projection of the data instances onto these principal axes are called the principal components. 

Then, we project our dataset using the formula:  
![](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-895a2e72455eb7165897c08c4c153196_l3.svg)

Dimensionality reduction is then obtained by only retaining those axes (dimensions) that account for most of the variance, and discarding all others.

![](https://media.geeksforgeeks.org/wp-content/uploads/20230420165637/Finding-Projection-in-PCA.webp)

Fig. Finding Projection in PCA

## Advantages of PCA

PCA (Principal Component Analysis) is like a data wizard 🧙‍♂️ that simplifies complex data with many benefits:

1. 📊 **Dimension Reduction**: It reduces the number of features, making data more manageable.
2. 💡 **Feature Selection**: It highlights the most important information.
3. 🌟 **Pattern Recognition**: It unveils hidden patterns and relationships.
4. 📈 **Data Visualization**: It helps plot data in a more understandable way.

For example, think of a colorful image. PCA turns it into a black and white sketch, preserving the main contours and shapes. 🖼️

## Limitations of PCA

1. Loss of Detail: PCA simplifies data, like turning a detailed painting into an abstract sketch. Imagine reducing a colorful photo of a sunset to black and white – you'd lose the vibrant hues 🌅.

2. Linearity Assumption: PCA assumes relationships are linear, but real-life data can be curvy. It's like expecting all your favorite songs to follow a straight beat, but some are jazzier 🎵.

3. Outliers Impact: Outliers can skew PCA results. Think of an outlier as that one friend who dances to a completely different rhythm at a party, throwing off your group's groove 💃.

4. Interpretability: While PCA simplifies, it can make results less intuitive. Imagine describing a complex novel using only a few emojis – you might miss some key plot details 📚🤔.

Example: Consider a dataset of mixed emotions from user reviews (happy, sad, and surprised). PCA might reduce them to "neutral" as the first principal component, losing the essence of each emotion 😐😢😲.

With the right feature reduction strategy, you'll have a streamlined dataset that powers your model efficiently, delivering accurate results without the headache of overfitting. 🏆

Get ready to conquer your data challenges! 🌟✨

For more details on PCA, click on the link.
[Step-by-Step Explanation of Principal Component Analysis](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

[Principal Component Analysis (PCA), Step-by-Step (Youtube Link)](https://www.youtube.com/watch?v=FgakZw6K1QQ&ab_channel=StatQuestwithJoshStarmer)

[Principle Component Analysis (PCA) Geometric Intuition (Youtube Link in Hindi)](https://www.youtube.com/watch?v=iRbsBi5W0-c&ab_channel=CampusX)

[PCA code and documentation are available here.](https://github.com/Avishek8136/Machine-Learning/blob/d949f6aba652a79b93281a9677f65bb33348fa2b/PCA.ipynb)