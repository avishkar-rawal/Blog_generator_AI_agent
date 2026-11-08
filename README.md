# Blog Generator AI Agent

## Project overview

This project is an agentic AI system that creates technical blog posts from a user-provided topic. It uses LangGraph to coordinate a multi-step workflow:

1. A router decides whether web research is needed.
2. The research node gathers and synthesizes sources with Tavily.
3. An orchestrator creates a structured blog plan.
4. Parallel workers write the individual blog sections.
5. A reducer combines the sections, decides whether technical images are useful, generates the images when needed, inserts the image links and captions into the Markdown, and saves the final blog.

The project also includes a Streamlit frontend for entering topics, watching graph progress, reviewing the plan and evidence, previewing the generated blog, viewing images, loading previous blogs, and downloading the results.

Generated Markdown files are saved in `generated_markdowns/`. Generated images are stored in its `images/` subdirectory and inserted into the blog using relative Markdown image links.

## Main technologies

- Python
- LangGraph
- LangChain
- Groq/OpenAI-compatible chat models
- Tavily web search
- Google Gemini image generation
- Streamlit
- Pydantic

## Resources

I followed these resources while learning and building this project:

- [YouTube tutorial](https://www.youtube.com/watch?v=Ou_v9lk0rxg&t=2251s)
- [CampusX blog-writing-agent repository](https://github.com/campusx-official/blog-writing-agent)
