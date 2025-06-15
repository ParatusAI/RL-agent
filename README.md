In the RL-agent repository there are 4 files. rl_agent_api.py is the api that gives the csbr and pbbr2 flow rates as well as the temperature. 
automation_agent.json is the n8n workflow that automates the process and calls on rl_agent_api.py. 
To use automatio_agent.json, upload prediction results of plqy, emission and fwhm from Isaiah's ML model as the input. The workflow will output ideal flow rates and temperature.  
sample_targets.csv is an sample of such predictions. Using the terminal upload the csv file to the workflow using:
curl -X POST \
     -F "file=@/Users/<your-username>/Downloads/sample_targets.csv;type=text/csv" \
     https://ryantrc.app.n8n.cloud/webhook-test/e31efaa2-2311-4cc1-bf99-25156bdd771d
this will initiate the workflow and output the flow rates and temperature. 

Make sure that rl_agent_api.py is running online. Then use ngrok to set it to a public server. From there the api can be freely called by the workflow. 





