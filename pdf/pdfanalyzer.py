from transformers import pipeline
import os

pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa", device_map='cuda')


job_description = """
    We are looking for a pragmatic Site Reliability Engineer who understands the nuances of production systems. If you care about building and running reliable software systems in production, you’ll like working at One2N.
    You will primarily work with our Startup and mid-size clients. We work on One-to-N kind problems (hence the name One2N), those where Proof of concept is done and the work revolves around scalability, maintainability and reliability.

    About You
       1. 2+ years of DevOps/SRE experience
       2. Good understanding of Linux systems and Bash scripting.
       3. Working knowledge of programming using Golang, Python, Java or equivalent
       4. Knowledge of IaaC tools (ex. Terraform) and Configuration management tools (ex. Ansible)
       5. Understanding of self hosted (ex. Jenkins) and cloud managed (GitLab CI, GitHub actions, etc) CI/CD tools
       6. Familiarity with Load balancers or Reverse proxies such as Nginx
       7. Understanding of networking (Subnets, CIDRs, NATs) and how internet works
       8. Experience with Infrastructure/Service observability using self hosted (ex. Elasticsearch, Logstash, Kibana or Prometheus, Loki, Grafana stack) tools or SaaS options like Datadog, NewRelic etc
       9. Knowledge of Cloud providers like AWS, GCP, Azure and IaaS, Saas, Paas, and FaaS services offered by these providers.
      10. Worked with Orchestrators like Kubernetes/Nomad/Docker Swarm.
      11. It’s a long list and it’s fine if you don’t have hands-on experience on some of these. We value curiosity over your current skill set. If you have strong CS fundamentals and are looking to get exposure to running reliable systems in production, let’s talk.
"""

prompt_for_suggestions = f"""
    Assume you are the HR representative responsible for looking over hiring operations at the company.
    The company is hiring for the role of a Site Reliability Engineer. 
    Given the Job Description for the role:
        {job_description}
    
    Please share your professional evaluation on whether the candidate's profile aligns with the role. 
    Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

prompt_for_candidate_background = """
    For the Given resume find the following details one at a time in the following order
    If a Field does not exist output NA
    {Full Name}
    {Address}
    {Phone number}
    {Email}
    {Total Experience (in years)}
"""

prompt_for_candidate_skills = """
    Generate the list of all skills from the resume. List each skill in one line
"""


prompt_for_ats_score=f"""
    You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
    your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
    the job description. First the output should come as percentage and then keywords missing and last final thoughts.
    The Job description :
        {job_description}
"""

data = pipe(
    os.path.join(os.curdir, 'out0.jpg'),
    prompt_for_suggestions,
    max_length=32 + len(prompt_for_suggestions)
)

print(data)