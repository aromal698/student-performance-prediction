st.header("🔐 Student Login")
st.write("Enter your student password to continue.")

password = st.text_input(
    "Password",
    type="password",
    placeholder="Example: BTECH2003"
)

if st.button("🔓 Login", type="primary", width="stretch"):

    password_clean = password.strip().upper()

    # Password must be BTECH + 4 digit year
    if (
        len(password_clean) == 9
        and password_clean[:5] == "BTECH"
        and password_clean[5:].isdigit()
        and 2000 <= int(password_clean[5:]) <= 2020
    ):
        st.session_state["student_logged_in"] = True
        st.session_state["student_dob_year"] = password_clean[5:]
        st.rerun()

    else:
        st.error(
            "Invalid password. Format must be BTECH + DOB year "
            "(2000–2020). Example: BTECH2003"
        )
