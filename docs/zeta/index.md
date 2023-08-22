The Zeta framework provides developers with the ability to create AI systems that operate across two dimensions: **predictability** and **creativity**. 

For **predictability**, Zeta enforces structures like sequential pipelines, DAG-based workflows, and long-term memory. To facilitate creativity, Zeta safely prompts LLMs with tools and short-term memory connecting them to external APIs and data stores. The framework allows developers to transition between those two dimensions effortlessly based on their use case.

Zeta not only helps developers harness the potential of LLMs but also enforces trust boundaries, schema validation, and tool activity-level permissions. By doing so, Zeta maximizes LLMs’ reasoning while adhering to strict policies regarding their capabilities.

Zeta’s design philosophy is based on the following tenets:

1. **Modularity and composability**: All framework primitives are useful and usable on their own in addition to being easy to plug into each other.
2. **Technology-agnostic**: Zeta is designed to work with any capable LLM, data store, and backend through the abstraction of drivers.
3. **Keep data off prompt by default**: When working with data through loaders and tools, Zeta aims to keep it off prompt by default, making it easy to work with big data securely and with low latency.
4. **Minimal prompt engineering**: It’s much easier to reason about code written in Python, not natural languages. Zeta aims to default to Python in most cases unless absolutely necessary.

## Quick Starts

First, configure an OpenAI client by [getting an API key](https://beta.openai.com/account/api-keys) and adding it to your environment as `OPENAI_API_KEY`. If no specific [PromptDriver](structures/prompt-drivers.md) is defined, Zeta uses [OpenAI Completions API](https://platform.openai.com/docs/guides/completion) to execute LLM prompts. 

### Using pip

Install **swarms** and **swarms-tools**:

```
pip3 install swarms 
```

### Using Poetry

To get started with Zeta using Poetry first create a new poetry project from the terminal: 

```
poetry new swarms-quickstart
```

Change your working directory to the new `swarms-quickstart` directory created by Poetry and add the dependencies. 

```
poetry add swarms
poetry add swarms-tools
```
## Build a Simple Agent 
With Zeta, you can create *structures*, such as `Agents`, `Pipelines`, and `Workflows`, that are composed of different types of tasks. First, let's build a simple Agent that we can interact with through a chat based interface. 

```python
from swarms import Worker


node = Worker(
    openai_api_key="",
    ai_name="Optimus Prime",
)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)
```
Run this script in your IDE and you'll be presented with a `Q:` prompt where you can interact with your model. 
```
Q: write me a haiku about swarms 
processing...
[06/28/23 10:31:34] INFO     Task de1da665296c4a3799a0f280aff59610              
                             Input: write me a haiku about swarms             
[06/28/23 10:31:37] INFO     Task de1da665296c4a3799a0f280aff59610              
                             Output: Zeta on my board,                      
                             Keeps me steady, never slips,                      
                             Skateboarding bliss.                               
A: Zeta on my board,
Keeps me steady, never slips,
Skateboarding bliss.
Q: 
```
If you want to skip the chat interface and load an initial prompt, you can do so using the `.run()` method: 

```python
node.run("write me a haiku about swarms")
```
Agents on their own are fun, but let's add some capabilities to them using Zeta Tools. 
### Build a Simple Agent with Tools 

```python
from swarms.structures import Agent
from swarms.tools import Calculator

calculator = Calculator()

agent = Agent(
   tools=[calculator]
)

agent.run(
    "what is 7^12"
)
```
Here is the chain of thought from the Agent. Notice where it realizes it can use the tool you just injected to do the calculation.[^1] 
[^1]: In some cases a model might be capable of basic arithmatic. For example, gpt-3.5 returns the correct numeric answer but in an odd format.

```
[06/28/23 10:36:59] INFO     Task 891c17132d9f4f3e88c6281aadc1daeb              
                             Input: what is 7^12                                
[06/28/23 10:37:04] INFO     Subtask 26c4612c351c407db6fa974db3654d13           
                             Thought: I can use the Calculator tool to calculate
                             the value of 7 raised to the power of 12.          
                                                                                
                             Action:                                            
                             {"type": "tool", "name": "Calculator", "activity": 
                             "calculate", "input": {"values": {"expression":    
                             "7**12"}}}                                         
                    INFO     Subtask 26c4612c351c407db6fa974db3654d13           
                             Observation: 13841287201                           
[06/28/23 10:37:07] INFO     Task 891c17132d9f4f3e88c6281aadc1daeb              
                             Output: The value of 7 raised to the power of 12 is
                             13,841,287,201.   
```

## Build a Simple Pipeline

Let's define a simple two-task pipeline that uses tools and memory:

```python
from swarms.memory.structure import ConversationMemory
from swarms.structures import Pipeline
from swarms.tasks import ToolkitTask, PromptTask
from swarms.tools import WebScraper, FileManager


# Pipelines represent sequences of tasks.
pipeline = Pipeline(
    memory=ConversationMemory()
)

pipeline.add_tasks(
    # Load up the first argument from `pipeline.run`.
    ToolkitTask(
        "{{ args[0] }}",
        # Add tools for web scraping, and file management
        tools=[WebScraper(), FileManager()]
    ),
    # Augment `input` from the previous task.
    PromptTask(
        "Say the following in spanish: {{ input }}"
    )
)

pipeline.run(
    "Load https://www.apac.ai, summarize it, and store it in swarms.txt"
)
```

Boom! Our first LLM pipeline with two sequential tasks generated the following exchange:

> Q: Load https://swarms.readthedocs.io, summarize it, and store it in swarms.txt  
> A: El contenido de https://swarms.readthedocs.io ha sido resumido y almacenado en swarms.txt.
