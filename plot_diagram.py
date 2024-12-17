import train_test

t = [t_items.item() for t_items in train_test.train_losses]
%matplotlib inline
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(t)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_test.train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(train_test.test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(train_test.test_acc)
axs[1, 1].set_title("Test Accuracy")