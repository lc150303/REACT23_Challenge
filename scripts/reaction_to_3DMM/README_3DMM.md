#### 1. get the emotion label from  .pkl

First , edit the `scripts/reaction_to_3DMM/read.py` 

```python
#change the input 
with open('results/finetune_100000_offline_s1_t0.33_k6_p1_5_samples.pkl', 'rb') as file:  
    data = pickle.load(file)
```

Then,bash the code

```bash
python read.py
```

 The emotion label (750,25) .csv file will be saved at   `scripts/reaction_to_3DMM/emotion` 

#### 2.get the 3DMM label from emotion label

We get 3DMM label use `scripts/reaction_to_3DMM/lstmtest.py` , you can just use our pre-trained model  `scripts/reaction_to_3DMM/lstm_model_200.pth`, the input and output folder can be changed there:

```python
model.load_state_dict(torch.load('lstm_model_200.pth'))

output_folder="scripts/reaction_to_3DMM/3DMMlabel"   #output 3DMM folder
folder_path = 'scripts/reaction_to_3DMM/emotion' 	#inout emotion folder
```

Then, bash the code 

```bash
python lstmtest.py
```

After that , we will get the 3DMM label (750,1,1,58) .npy  saved at  `scripts/reaction_to_3DMM/3DMMlabel`



#### 3. generate the Visualization from 3DMM label

We generate the Visualization use `scripts/reaction_to_3DMM/render_all.py` 

 the input and output folder can be changed there:

```python
 folder='scripts/reaction_to_3DMM/3DMMlabel'     #load 3dmmlabel 
 output_path="scripts/reaction_to_3DMM/visual_output"   #the Visualization will output in this folder
```



IMPORTANT,IMPORTANT,IMPORTANT!!!!!!

you should  screenshot the listener_frame  include frontal face  with 256*256 size , and put it in   `scripts/reaction_to_3DMM/listenner_frame`  and modify  the code of load the listenner_frame.

```python
if '152153' in file_path:
            listener_reference = pil_loader('listenner_frame/152153.png')      
elif 'RECOLA' in file_path:
            listener_reference = pil_loader('listenner_frame/recola.png')
elif '001' in file_path:
            listener_reference = pil_loader('listenner_frame/001.png')
elif '019' in file_path:
            listener_reference = pil_loader('listenner_frame/019.png')
else:
            listener_reference = pil_loader('listenner_frame/023.png')
```

the '152153' , 'RECOLA' , '001', '019'   key-word should be modified to  the  only choice to match the 3DMM and the listenner_frame

Then, bash the code

```bash
python render_all.py
```

The Visualization will be saved at  `scripts/reaction_to_3DMM/visual_output` 