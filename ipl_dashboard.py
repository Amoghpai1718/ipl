# ======================================================
# TAB 2: Simple AI Chatbot
# ======================================================
with tab2:
    st.subheader("💬 Ask the IPL Chatbot")

    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    GOOGLE_CX = st.secrets.get("GOOGLE_SEARCH_CX")

    if not GOOGLE_API_KEY or not GOOGLE_CX:
        st.warning("Chatbot disabled. Add GOOGLE_API_KEY and GOOGLE_SEARCH_CX in Streamlit Secrets.")
    else:
        user_query = st.text_input("Ask anything about IPL...")
        if st.button("Ask"):
            try:
                url = f"https://www.googleapis.com/customsearch/v1?q={user_query}&cx={GOOGLE_CX}&key={GOOGLE_API_KEY}"
                resp = requests.get(url).json()
                if "items" in resp:
                    for item in resp["items"][:3]:
                        st.write(f"**{item['title']}** - {item['snippet']}")
                else:
                    st.write("No relevant results found.")
            except Exception as e:
                st.error(f"Chatbot error: {e}")
