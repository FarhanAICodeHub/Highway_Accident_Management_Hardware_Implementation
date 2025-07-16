import openai

# Initialize OpenAI API key
openai.api_key =  ""
# Path to the .txt file 
file_path = "/home/hasan/drone_p3_implementation/image_descriptions.txt"


# Read descriptions from the file
with open(file_path, "r") as file:
    descriptions = file.readlines()

# Function to generate a two-line summary for all descriptions collectively
def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", #"gpt-4-turbo",  # Specify gpt-4-turbo for cost-effective usage
            messages=[#{"role": "system", "content": "You are an expert accident response supervisor."},
                {"role": "user", "content": f"This is related to highway accident response. You'll be provided accident descriptions(around 10, same accident).\nYou're acting as the supervisor, That'll first describe the accident scene in two lines(This contains accident alert level and summary) and then guide the team for suggested health/emergency related responses(max two lines, these can be related to rescue, ambulance, disaster or police team).Description text from accident is :\n\n{text}"}],
            max_tokens=200,  # Limit tokens to keep it concise        
            temperature=0.3,  # Controlled randomness
            top_p=1,  # Use full probability distribution
            frequency_penalty=0.2,  # Avoid word repetition
            presence_penalty=0.2  # Encourage new ideas
        )
        summary = response.choices[0].message["content"]
        return summary.strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

# Generate the collective summary
collective_summary = summarize_text(descriptions)

# Save the collective summary to a new .txt file
with open("collective_summary.txt", "w") as output_file:
    output_file.write(f"Collective Summary:\n{collective_summary}")

print("Collective summary saved to collective_summary.txt")
