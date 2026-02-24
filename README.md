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

<img width="578" height="465" alt="image" src="https://github.com/user-attachments/assets/6011299a-2090-47d9-bc3f-44004fae523b" />


ViViT pretrained on WLASL-100 with 30 epochs and early stopping:

Stopped at epoch 14

Per-epoch training time = 342s


<img width="569" height="443" alt="image" src="https://github.com/user-attachments/assets/d6a27cf9-81e1-4a01-a807-2008483b3ac7" />

# Results on WLASL-300 with full fine-tuning
SlowFast pretrained on Kinetic-400 with 30 epochs and early stopping:

Stopped at epoch 19

Per-epoch training time = 858s

Test accuracy = 38.77%


<img width="519" height="420" alt="image" src="https://github.com/user-attachments/assets/3644403a-4864-4397-9486-85fe43b3c6ee" />


ViViT pretrained on WLASL-100 with 30 epochs and early stopping:

Stopped at epoch 22

Per-epoch training time = 1737s


<img width="584" height="467" alt="image" src="https://github.com/user-attachments/assets/09fe6317-1653-4d72-ae68-ee3b5661c22f" />

# Results on WLASL-100 with LoRA
SlowFast pretrained on Kinetic-400 with 30 epochs and early stopping:

Stopped at epoch 26

Per-epoch training time = 310s

Test accuracy = 46.12%


<img width="522" height="419" alt="image" src="https://github.com/user-attachments/assets/e49a82dd-0a70-4284-b32c-a7ee545812ba" />

