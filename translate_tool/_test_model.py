import os
os.environ['TOKENHUB_API_KEY'] = 'sk-oXR2tNcqBPs3vS0GYeqqrk719f36x6tDzKJYsaI'
import openai
client = openai.OpenAI(api_key=os.environ['TOKENHUB_API_KEY'], base_url='https://tokenhub.tencentmaas.com/v1')
# Test 1: plain
try:
    r = client.chat.completions.create(model='deepseek-v4-flash-202605', messages=[{'role':'user','content':'Say hi in Chinese'}], max_tokens=30)
    print('PLAIN OK:', repr(r.choices[0].message.content))
except Exception as e:
    print('PLAIN FAIL:', str(e)[:200])
# Test 2: with system + json instruction
try:
    r = client.chat.completions.create(
        model='deepseek-v4-flash-202605',
        temperature=0.3,
        messages=[
            {'role':'system','content':'You are a translator. Return JSON.'},
            {'role':'user','content':'Translate to Chinese (return JSON array): ["Hello", "World"]'},
        ],
        max_tokens=80,
    )
    print('PROMPT OK:', repr(r.choices[0].message.content))
except Exception as e:
    print('PROMPT FAIL:', str(e)[:200])
# Test 3: with response_format json_object
try:
    r = client.chat.completions.create(
        model='deepseek-v4-flash-202605',
        temperature=0.3,
        messages=[
            {'role':'system','content':'You are a translator. Return JSON.'},
            {'role':'user','content':'Translate ["Hello", "World"] to Chinese. Return as JSON: {"translations": [...]}'},
        ],
        max_tokens=80,
        response_format={'type':'json_object'},
    )
    print('JSON OK:', repr(r.choices[0].message.content))
except Exception as e:
    print('JSON FAIL:', str(e)[:200])
