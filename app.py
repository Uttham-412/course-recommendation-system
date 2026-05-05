import streamlit as st
from model import recommend

st.set_page_config(page_title="Course Recommender", page_icon="📚")

st.title("📚 Course Recommendation System")

course = st.text_input("What do you want to learn? (e.g., python, web development, finance)")

if st.button("Recommend"):
    if course.strip() == "":
        st.warning("Please enter a topic or keyword.")
    else:
        results = recommend(course)

        st.subheader("Recommended Courses:")

        if not results:
            st.error("No similar courses found. Try different keywords.")
        else:
            for r in results:
                # Show clickable link and metadata
                title = r['title']
                url = r['url']
                subject = r['subject']
                level = r['level']
                price = r['price']
                
                # Use st.markdown to create a clickable link
                st.markdown(f"### 👉 [{title}]({url})")
                st.write(f"**Subject:** {subject} | **Level:** {level} | **Price:** ${price}")
                st.write("---")