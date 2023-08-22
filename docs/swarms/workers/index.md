### Enterprise Grade Documentation

---

## Worker Class

The `Worker` class represents a core component of the framework designed to interact with ChatOpenAI models, utilize embeddings and vector stores, and run tasks. Enhanced with logging, error handling, and timing utilities, this class is integral for scalable and efficient task processing.

```
from swarms import WorkerNode
```

---

### Constructor: `Worker.__init__()`

**Description**:  
Initializes the `Worker` object by setting up necessary configurations, tools, memory, and the agent.

**Parameters**:
- `model_name (str)`: Name of the ChatOpenAI model. Default is "gpt-4".
- `openai_api_key (str)`: API Key for OpenAI services. Default is `None`.
- `ai_name (str)`: Name for the AI worker. Default is "Autobot Swarm Worker".
- `ai_role (str)`: Role of the AI worker. Default is "Worker in a swarm".
- `temperature (float)`: Temperature setting for the ChatOpenAI model, affecting randomness. Default is `0.5`.

**Returns**:
- None

**Example Usage**:
```python
worker = Worker(model_name="gpt-4", openai_api_key="YOUR_API_KEY")
```

---

### Method: `Worker.setup_tools()`

**Description**:  
Sets up essential tools required by the worker for processing tasks. 

**Parameters**:
- None

**Returns**:
- None

**Example Usage**:
```python
# Typically called internally during Worker initialization
worker.setup_tools()
```

---

### Method: `Worker.setup_memory()`

**Description**:  
Initializes the embeddings and vector stores required for task processing.

**Parameters**:
- None

**Returns**:
- None

**Example Usage**:
```python
# Typically called internally during Worker initialization
worker.setup_memory()
```

---

### Method: `Worker.setup_agent()`

**Description**:  
Initializes the agent to use the provided tools and memory setups.

**Parameters**:
- None

**Returns**:
- None

**Example Usage**:
```python
# Typically called internally during Worker initialization
worker.setup_agent()
```

---

### Method: `Worker.run(task)`

**Description**:  
Executes the specified task using the agent.

**Parameters**:
- `task`: The task to be executed by the agent.

**Returns**:
- `result`: The output after executing the task.

**Example Usage**:
```python
task = "Some specific task"
result = worker.run(task)
print(result)
```

---

### Method: `Worker.__call__(task)`

**Description**:  
Allows the worker instance to be called as a function to execute the given task.

**Parameters**:
- `task`: The task to be executed by the agent.

**Returns**:
- `results`: The output after executing the task.

**Example Usage**:
```python
from swarms import Worker

worker_instance = Worker()
task = "Some specific task"
result = worker_instance(task)  # Using worker as callable
print(result)
```
**Configured Worker**:
```python
from swarms import Worker


node = Worker(
    openai_api_key="",
    ai_name="Optimus Prime",
    ai_role="You are John a nice worker in a swarm",
    model_name="gpt-4
)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)

```


---

**Note**:
1. Ensure you have the necessary dependencies installed and the OpenAI API key properly configured before initializing the `Worker` class.
2. Handle exceptions gracefully in production environments.
3. Always refer to the latest official documentation for updates and more details.

