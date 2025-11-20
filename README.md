# Google Michigan Dev Fest-2025
In this workshop we will explore together the essential methodology for visualizing the results of neural network training runs using PyTorch’s integration with TensorBoard. While simple statistics can be printed during training, TensorBoard offers a superior tool for creating interactive visualizations. 
	
We will be using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist/tree/master/data), detail the setup, including defining a SummaryWriter, and demonstrate five key areas of visualization:
1. Inspecting Model Architecture: We will show how to visualize complex model structures via the Graphs tab using $writer.addgraph()$.
2. Visualizing High-Dimensional Data: We will demonstrate the utility of the Projector tab by using the add_embedding method to project high-dimensional image data into lower dimensional space for interactive exploration.
3. Tracking Training Progression: We will illustrate effective methods for logging the running loss during training using add_scalar.
4. Monitoring Intermediate Predictions: We will demonstrate how to log Matplotlib Figures showing the model's predictions versus actual labels on batches throughout the training process.
5. Assessing Trained Models: Finally, we will cover methods for post-training assessment by plotting precision-recall curves for each class, providing detailed insights into per-class accuracy and performance.
	
This session will provide audience with the practical steps needed to move beyond console output, ensuring a clearer and more insightful understanding of model performance and training progression.
This session is inspired from the [PyTorch Intermediate Learning Session](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html), from the official PyTorch website, and I have just put all the pieces together in a Jupyter Notebook for ease of use and further exploration. 


