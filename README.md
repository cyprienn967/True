Okay so currently what happens when everything is up and running is the following:
Main page (most buttons don't work yet).
Sign in button on top right.
Clicking it brings you to a sign in page, need right username and password for it to work. 
If correct, takes you to page
where previous user 'metrics' are there, and option to paste in API key. 
If you press test, API key gets sent to the backend, backend runs test_chat_gpt.py which gives an average of prompt response lengths and it displays it on the dashboard.

Structure:
1. Backend - app.py and test_chat_gpt.py basically take in API key and perform stuff on it (the latter sends in 10 prompts, collects 50 responses and averages the length of the responses). Server.js basically takes in the API key from the frontend and passes it to the backend. Other files not important
2. frontend - App.js contains links to the files for all the other pages pretty much. Dashboard.js is the dashboard once you've signed in, MainPage.js, SignIn.js similar. ProtectedRoute.js is basically MEANT to make it so that, since the dashboard is at like websitename.com/dashboard, you can't access it by just typing it into the browser without signing in first although i dont think it currently works - to be fixed later.
3. semantic_uncertainty1 - code from a repo of a paper on semantic uncertainity - needs adapting for dynamic API stuff once we've got pipeline and frontend fully working
4. testing - any python scripts for testing stuff like calls, syntax etc (stuff which doesn't rely on frontend/backend info)

To run this all yourself gonna need to install some packages and shit:
off the top of my head (this maybe isnt the completed list but when you try and run it all it'll say whats not installed)
pip install openai
pip install flask
uhhh something for the frontend

To run: main page with sign in and authentication and takes in API key and does tests on it
[/backend] python app.py
[/W/E] npm run dev
click on the localhost link in the terminal
wait like 5 seconds


To do:
Make protectedroute.js
finish building website frontend i.e. make template specific to us for the rest of it
figure out secure way of storing data (external database) and then link it all back up
then, can start writing bias/hallucination/jailbreak/CoT/false-alignments



