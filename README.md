# seniorproject_signlanguage
This is a repository for hosting CSCI 409 Senior Project documentation and files

# Results with no fine-tuning for WLASL-100
SlowFast pretrained on Kinetic-400:
Accuracy = 7.8%

ViViT pretrained on WLASL-100:
Accuracy = 27.5%

# Results with no fine-tuning for WLASL-300
SlowFast pretrained on Kinetic-400:
Accuracy = 1.5%

ViViT pretrained on WLASL-100:
Accuracy = 0.2%

# Results on WLASL-100 with full fine-tuning
SlowFast pretrained on Kinetic-400 with 30 epochs and early stopping:
Stopped at epoch 17
Per-epoch training time = 701.4s
Test accuracy = 54.26%

ViViT pretrained on WLASL-100 with 30 epochs and early stopping:
Stopped at epoch 14
Per-epoch training time = 342s

# Results on WLASL-300 with full fine-tuning
SlowFast pretrained on Kinetic-400 with 30 epochs and early stopping:
Stopped at epoch 19
Per-epoch training time = 858s
Test accuracy = 38.77%

ViViT pretrained on WLASL-100 with 30 epochs and early stopping:
Stopped at epoch 22
Per-epoch training time = 1737s

# Results on WLASL-100 with LoRA
SlowFast pretrained on Kinetic-400 with 30 epochs and early stopping:
Stopped at epoch 26
Per-epoch training time = 310s
Test accuracy = 46.12%

