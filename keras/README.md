# how to use with the model on GCP:


### prerequisites"
* you need the google cloud SDK install on your machine first.
* uncomment the lines in `resNet50.py` and run it.


create new bucket:
`gsutil mb -l <region> gs://[your-bucket-name]`

copy local files to the bucket:  
`gsutil cp -R <path-to-local-folder> gs://[your-bucket-name]  `  

create model:  
`gcloud ml-engine models create [model-name] --region <region>`   

create new version:  
`gcloud ml-engine versions create v1 --model=[model-name] --origin=gs://[your-bucket-name]`  

run the commend:  
`gcloud ml-engine predict --model=[earnings/resnet50] --json-instances=<path-to-json-for-prediction> `


## to use the model from code :

* you need to crete a service_account credentioal file from your GCP console on:  APIs & Services > credentials > crete credentials > service account key 
* download the JSON file
* change variables:  
  * CREDENTIALS_FILE - path to your json file.  
  * PROJECT_ID - your project id name.
  * MODEL_NAME - your model name.  
    
