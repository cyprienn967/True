Okay so currently what happens when everything is up and running is the following:
On the website there is a textbox to paste in an LLM key
LLM key gets sent back to backend which uses it to prompt the LLM and get responses
responses are sent back to the frontend and displayed on website

Structure:
1. Backend - just contains the app.py script that does the prompting stuff (rest is just flask files idk what they do ngl)
2. frontend - main stuff for website in frontend/src/App.js its all in react (all the other files are just react preloaded files lowkey not important)
3. semantic_uncertainty1 - code from a repo of a paper on semantic uncertainity - needs adapting for dynamic API stuff once we've got pipeline and frontend fully working
4. testing - any python scripts for testing stuff like calls, syntax etc.

