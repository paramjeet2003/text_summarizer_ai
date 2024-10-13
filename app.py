import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

##  get the groq api key and url field to be summarize
with st.sidebar:
    api_key=st.text_input("Groq API Key",value="",type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## gemma model using groq api
if not api_key:
    st.error("Please provide a valid Groq API key.")
else:
    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=api_key)


## prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


final_prompt="""
Provide all the important points.
incorporating the additional information like the names involved 
in the video or on the website if any and , 
Start the summary with an introduction and
then provide a fantastic summary
Speech:{text}
Summary:
Refine the summary,  like all the
names involved  and give a fantastic summary
"""

final_prompt_template = PromptTemplate(
    input_variables=['text'],
    template=final_prompt
)

if st.button("Summarize the Content from YT ot Website"):
    if not api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("URL given is not valid")
    else:
        try:
            with st.spinner("Waiting...."):
                ## loading the YT or website video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(
                        urls=[generic_url], 
                        ssl_verify=False, 
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
                split = text_splitter.split_documents(docs)

                ## chain for summarizarion
                chain = load_summarize_chain(
                    llm=llm, 
                    chain_type="map_reduce", 
                    map_prompt=prompt,
                    combine_prompt=final_prompt_template
                )
                output_summary= chain.run(split)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")

