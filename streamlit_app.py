import streamlit as st
import time
import requests


def main():
    st.set_page_config(  # Alternate names: setup_page, page, layout
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title="The Big Language Model Workshop",  # String or None. Strings get appended with "• Streamlit".
        page_icon=None,  # String, anything supported by st.image, or None.
    )
    st.title("kogpt 6B playground")
    """kogpt 6B playground"""

    ex_names = ["""인간처럼 생각하고, 행동하는 \'지능\'을 통해 인류가 이제까지 풀지 못했던""",
    ]
    example = st.selectbox("Choose an example prompt from this selector", ex_names)

    inp = st.text_area(
        "Or write your own prompt here!", example, max_chars=2000, height=150
    )

    try:
        rec = ex_names.index(inp)
    except ValueError:
        rec = 0

    with st.beta_expander("Generation options..."):
        length = st.slider(
            "Choose the length of the generated texts (in tokens)",
            2,
            1024,
            64 if rec < 2 else 50,
            10,
        )
        temp = st.slider(
            "Choose the temperature (higher - more random, lower - more repetitive). For the code generation or sentence classification promps it's recommended to use a lower value, like 0.35",
            0.0,
            1.5,
            1.0 if rec < 2 else 0.35,
            0.05,
        )

    response = None
    with st.form(key="inputs"):
        submit_button = st.form_submit_button(label="Generate!")

        if submit_button:

            payload = {
                "context": inp,
                "token_max_length": length,
                "temperature": temp,
                "top_p": 0.9,
            }

            query = requests.post("http://localhost:5000/generate", params=payload)
            response = query.json()

            st.markdown(response["prompt"] + response["text"])
            st.text(f"Generation done in {response['compute_time']:.3} s.")

    if False:
        col1, col2, *rest = st.beta_columns([1, 1, 10, 10])

        def on_click_good():
            response["rate"] = "good"
            print(response)

        def on_click_bad():
            response["rate"] = "bad"
            print(response)

        col1.form_submit_button("👍", on_click=on_click_good)
        col2.form_submit_button("👎", on_click=on_click_bad)

    st.text("by @noah")


if __name__ == "__main__":
    main()