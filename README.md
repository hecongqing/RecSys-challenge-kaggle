# RecSys-challenge-kaggle
 Given a sequence of click events performed by some user during a typical session in an e-commerce website, the goal is to predict whether the user is going to buy something or not, and if he is buying, what would be the items he is going to buy. The task could therefore be divided into two sub goals: 
 
 1. Is the user going to buy items in this session? 
 2. If yes, what are the items that are going to be bought?

Divide the data (clicks and buys) into two parts based on the timestamp---training (roughly 2/3rds of dataset based on timestamp) and test data. Train your model on the training data and evaluate its performance on the test data.

**The data comprises two different files:**

yoochoose-clicks.dat - Click events. 
Each record/line in the file has the following fields:
1. Session ID – the id of the session. In one session there are one or many clicks.
2. Timestamp – the time when the click occurred.
3. Item ID – the unique identifier of the item.
4. Category – the category of the item.

yoochoose-buys.dat - Buy events. 
Each record/line in the file has the following fields:
1. Session ID - the id of the session. In one session there are one or many buying events.
2. Timestamp - the time when the buy occurred.
3. Item ID – the unique identifier of item.
4. Price – the price of the item.
5. Quantity – how many of this item were bought.

The Session ID in yoochoose-buys.dat will always exist in the yoochoose-clicks.dat file – the records with the same Session ID together form the sequence of click events of a certain user during the session. The session could be short (few minutes) or very long (few hours), it could have one click or hundreds of clicks. All depends on the activity of the user.

Evaluation Measure used in the challenge
The evaluation is taking into consideration the ability to predict both aspects – whether the sessions end with buying event, and what were the items that have been bought. Let’s define the following (Note that all capital letters represent sets.):
- Sl – sessions in submitted solution file
- S - All sessions in the test set
- s – session in the test set
- Sb – sessions in test set which end with buy o As – predicted bought items in session s
- Bs – actual bought items in session s
then the score of a solution will be :

![Screen Shot 2022-06-24 at 11 59 52 AM](https://user-images.githubusercontent.com/70657426/175521666-ccef9219-5751-4224-bec9-5b27460220ec.png)


