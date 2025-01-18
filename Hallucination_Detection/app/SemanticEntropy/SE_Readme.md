To install necessary dependencies, run the following command:
```conda-env update -f environment.yaml
conda activate semantic_uncertainty
```

To start the API run the flask script in
```semantic_uncertainty/SE_api_script.sh```
It will likely run in port 5000, so you can access the API at http://localhost:5000/

The API uses OpenAI API so you need to have an API key. You can get one at https://beta.openai.com/signup/

The API has one endpoint, /api/data, which receives a GET request with a JSON body containing the prompt that you want to run. The output is then analyzed using semantic entropy, which will be outputted. The JSON body should have the following format:
```
{
  "id": placeholder,
  "api_key": "your_openai_key",
  "prompt": "The text you want to analyze"
  "inference_temperature": 1.0,

}
```

The API is found in the file ```semantic_uncertainty/SE_api.py```.