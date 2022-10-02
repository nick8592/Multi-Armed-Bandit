# Multi-Armed-Bandit

As figure shown below, 
Thompson Sampling has the the highest mean total reward, second UCB and at last Epsilon-Greedy. 
Before 600 episodes, we can see that UCB’s reward is a little bit lower than Epsilon-Greedy. 
Because UCB prefer actions that it hasn’t had a confident value estimation for it yet. 
After enough exploration than UCB will focus on the one which has strong potential 
to have a optimal value. 
Therefore after 600 episodes, UCB gradually get better results than Epsilon-Greedy.

![Average Reward Comparison](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4690134a-083f-41e5-9fb2-935b27e33f74/Figure_1.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221002T063825Z&X-Amz-Expires=86400&X-Amz-Signature=7721fc8b21d2627c49e10401e25c553c36065e145839be749f963cff9f2b8805&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Figure_1.png%22&x-id=GetObject)

As figure shown below, 
Epsilon-Greedy spent the shortest time, second UCB and Thompson Sampling spent the longest time.
According to the result, more complex algorithm may get better result, 
but it may also cost more time on calculation. 
It’s a trade-off between “Algorithm Complexity” and “Time Consumption” for developer.
![Time Comparison](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3c7992a5-8796-46f8-9a29-9d2836b38e59/Figure_2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221002T063851Z&X-Amz-Expires=86400&X-Amz-Signature=cd30b60a04248715c08729e3718df0426ca1f9921bd7849ca0cf54f8ca60cd4b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Figure_2.png%22&x-id=GetObject)
