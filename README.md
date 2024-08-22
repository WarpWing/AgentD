# AgentD
A custom [ReAct Agent](https://arxiv.org/abs/2210.03629) designed to assist with Dickinson related inquires. Has an focus on sustainability and is a successor to [Gaia](https://github.com/WarpWing/GreenQuestChat/tree/main). The full version of "AgentD" should eventually come with a RAG tool (FAISS), Agent Tools (For Faculty Search and broader Web Search grounded on Dickinson's domain with an eventual LFU caching implementation.) 

Demo: https://agentd.warpwing.cloud/

## To Do List
- [ ] Implement final answer streaming
- [ ] Implement FAISS properly
- [ ] Fix the semantic and document cache (as well as set expiry)
- [ ] Streamline document handling and chunking
- [ ] Streamline prompt to use template reason (set required information for a complete answer)
- [ ] Fix the output demo to properly show all agent actions
- [ ] Cut down on generation time, possibly look into asynchrous functional calling with agent (multiple tool calls at once)
- [ ] (Optional): Possible implement a backend service that handles the update of documents
