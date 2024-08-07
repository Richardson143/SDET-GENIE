
import streamlit as st
import os, csv, base64, io, time, json
from PIL import Image
from streamlit_lottie import st_lottie
import requests
from io import BytesIO
from llama_index.core.llms import ChatMessage
from selenium import webdriver
from lavague.core import WorldModel, ActionEngine
from lavague.core.agents import WebAgent
from lavague.core.context import Context
from lavague.drivers.selenium import SeleniumDriver
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from typing import List, Tuple, Dict
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Initialize the LLM and other required components
llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=os.getenv("GOOGLE_API_KEY"))
mm_llm = GeminiMultiModal(model_name="models/gemini-1.5-pro-latest", api_key=os.getenv("GOOGLE_API_KEY"))
embedding = GeminiEmbedding(model_name="models/text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY"))

context = Context(llm=llm, mm_llm=mm_llm, embedding=embedding)

def main(url, feature_content, language):
    # Parse feature content
    feature_name = "generated_feature"
    feature_file_name = f"{feature_name}.feature"
    test_case = feature_content
    # Initialize the agent
    selenium_driver = SeleniumDriver(headless=False)
    world_model = WorldModel.from_context(context)
    action_engine = ActionEngine.from_context(context, selenium_driver)
    agent = WebAgent(world_model, action_engine)
    objective = f"Run this test case: \n\n{test_case}"
    # Run the test case with the agent
    print("--------------------------")
    print(f"Running test case:\n{test_case}")
    agent.get(url)
    agent.run(objective)
    # Perform RAG on final state of HTML page using the action engine
    print("--------------------------")
    print(f"Processing run...\n{test_case}")
    nodes = action_engine.navigation_engine.get_nodes(
        f"We have ran the test case, generate the final assert statement.\n\ntest case:\n{test_case}"
    )
    # Parse logs
    logs = agent.logger.return_pandas()
    last_screenshot_path = get_latest_screenshot_path(logs.iloc[-1]["screenshots_path"])
    b64_img = pil_image_to_base64(last_screenshot_path)
    selenium_code = "\n".join(logs["code"].dropna())
    print("--------------------------")
    print(f"Generating {language} code")
    # Generate test code
    if language.lower() == "python":
        code = generate_pytest_code(url, feature_file_name, test_case, selenium_code, nodes, b64_img)
    else:  # Java
        code = generate_java_code(url, feature_file_name, test_case, selenium_code, nodes, b64_img)
    code = code.replace("```python", "").replace("```java", "").replace("```", "").replace("```\n", "").strip()
    # Write test code to file
    file_extension = "py" if language.lower() == "python" else "java"
    output_path = f"./tests/{feature_name}.{file_extension}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("--------------------------")
    print(
        f"{language} test code for feature: {feature_name} has been generated in {output_path}"
    )
    return output_path, code

def get_latest_screenshot_path(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Get the full path of the files
    full_paths = [os.path.join(directory, f) for f in files]
    # Find the most recently modified file
    latest_file = max(full_paths, key=os.path.getmtime)
    return latest_file

def pil_image_to_base64(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert image to BytesIO object
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        # Encode the BytesIO object to base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_pytest_code(url, feature_file_name, test_case, selenium_code, nodes, b64_img):
    prompt = f"""Generate a Python Selenium test script with the following inputs and structure examples to guide you:
    Base url: {url}
    Feature file name: {feature_file_name}
    Test case: {test_case}
    Already executed code:
    {selenium_code}
    Selected html of the last page: {nodes}
    Image: {b64_img}
    Examples:
    {PYTHON_EXAMPLES}
    """
    messages = [ChatMessage(role="user", content=prompt)]
    response = llm.chat(messages)
    return response.message.content

def generate_java_code(url, feature_file_name, test_case, selenium_code, nodes, b64_img):
    prompt = f"""Generate a Java Selenium test script with the following inputs and structure examples to guide you:
    Base url: {url}
    Feature file name: {feature_file_name}
    Test case: {test_case}
    Already executed code:
    {selenium_code}
    Selected html of the last page: {nodes}
    Image: {b64_img}
    Examples:
    {JAVA_EXAMPLES}
    """
    messages = [ChatMessage(role="user", content=prompt)]
    response = llm.chat(messages)
    return response.message.content

PYTHON_EXAMPLES = """
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Constants
BASE_URL = '{url}'

class JobApplicationTest:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(10)

    def setup(self):
        self.driver.get(BASE_URL)

    def teardown(self):
        self.driver.quit()

    def given_i_am_on_the_job_application_page(self):
        # This step is handled by the setup method
        pass

    def when_i_enter_first_name(self, first_name):
        first_name_field = self.driver.find_element(By.XPATH, "/html/body/form/div[1]/ul/li[2]/div/div/span[1]/input")
        first_name_field.send_keys(first_name)

    def when_i_enter_last_name(self, last_name):
        last_name_field = self.driver.find_element(By.XPATH, "/html/body/form/div[1]/ul/li[2]/div/div/span[2]/input")
        last_name_field.send_keys(last_name)

    def when_i_enter_email_address(self, email):
        email_field = self.driver.find_element(By.XPATH, "/html/body/form/div[1]/ul/li[3]/div/span/input")
        email_field.send_keys(email)

    def when_i_enter_phone_number(self, phone_number):
        phone_number_field = self.driver.find_element(By.XPATH, "/html/body/form/div[1]/ul/li[4]/div/span/input")
        phone_number_field.send_keys(phone_number)

    def when_i_leave_cover_letter_empty(self):
        # No action needed as the field should remain empty
        pass

    def when_i_click_apply_button(self):
        apply_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "/html/body/form/div[1]/ul/li[6]/div/div/button"))
        )
        self.driver.execute_script("arguments[0].scrollIntoView(true);", apply_button)
        apply_button.click()

    def then_i_should_see_error_message_for_cover_letter(self):
        try:
            error_message = self.driver.find_element(By.XPATH, "/html/body/form/div[1]/ul/li[5]/div/div/span")
            assert error_message.is_displayed(), "Error message for Cover Letter field is not displayed"
        except Exception as e:
            raise AssertionError(f"Error message not displayed: {e}")

    def run_test(self):
        try:
            self.setup()
            self.given_i_am_on_the_job_application_page()
            self.when_i_enter_first_name("John")
            self.when_i_enter_last_name("Doe")
            self.when_i_enter_email_address(john.doe@example.com)
            self.when_i_enter_phone_number("(123) 456-7890")
            self.when_i_leave_cover_letter_empty()
            self.when_i_click_apply_button()
            self.then_i_should_see_error_message_for_cover_letter()
            print("Test passed: Job application scenario completed successfully.")
        except AssertionError as e:
            print(f"Test failed: {str(e)}")
        except Exception as e:
            print(f"Test failed: An unexpected error occurred: {str(e)}")
        finally:
            self.teardown()

if __name__ == "__main__":
    test = JobApplicationTest()
    test.run_test()
"""

JAVA_EXAMPLES = """
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.testng.Assert;
import org.testng.annotations.AfterMethod
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

public class JobApplicationTest {
    private WebDriver driver;
    private String baseUrl;

    @BeforeMethod
    public void setup() {
        System.setProperty("webdriver.chrome.driver", "/path/to/chromedriver");
        driver = new ChromeDriver();
        driver.manage().timeouts().implicitlyWait(10, java.util.concurrent.TimeUnit.SECONDS);
        baseUrl = "{url}"; // Replace with the actual base URL
        driver.get(baseUrl);
    }

    @AfterMethod
    public void teardown() {
        driver.quit();
    }

    @Test
    public void jobApplicationScenario() {
        givenIAmOnTheJobApplicationPage();
        whenIEnterFirstName("John");
        whenIEnterLastName("Doe");
        whenIEnterEmailAddress(john.doe@example.com);
        whenIEnterPhoneNumber("(123) 456-7890");
        whenILeaveCoverLetterEmpty();
        whenIClickApplyButton();
        thenIShouldSeeErrorMessageForCoverLetter();
    }

    private void givenIAmOnTheJobApplicationPage() {
        // This step is handled by the setup method
    }

    private void whenIEnterFirstName(String firstName) {
        WebElement firstNameField = driver.findElement(By.xpath("/html/body/form/div[1]/ul/li[2]/div/div/span[1]/input"));
        firstNameField.sendKeys(firstName);
    }

    private void whenIEnterLastName(String lastName) {
        WebElement lastNameField = driver.findElement(By.xpath("/html/body/form/div[1]/ul/li[2]/div/div/span[2]/input"));
        lastNameField.sendKeys(lastName);
    }

    private void whenIEnterEmailAddress(String email) {
        WebElement emailField = driver.findElement(By.xpath("/html/body/form/div[1]/ul/li[3]/div/span/input"));
        emailField.sendKeys(email);
    }

    private void whenIEnterPhoneNumber(String phoneNumber) {
        WebElement phoneNumberField = driver.findElement(By.xpath("/html/body/form/div[1]/ul/li[4]/div/span/input"));
        phoneNumberField.sendKeys(phoneNumber);
    }

    private void whenILeaveCoverLetterEmpty() {
        // No action needed as the field should remain empty
    }

    private void whenIClickApplyButton() {
        WebElement applyButton = new WebDriverWait(driver, 10).until(
                ExpectedConditions.elementToBeClickable(By.xpath("/html/body/form/div[1]/ul/li[6]/div/div/button"))
        );
        ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", applyButton);
        applyButton.click();
    }

    private void thenIShouldSeeErrorMessageForCoverLetter() {
        try {
            WebElement errorMessage = driver.findElement(By.xpath("/html/body/form/div[1]/ul/li[5]/div/div/span"));
            Assert.assertTrue(errorMessage.isDisplayed(), "Error message for Cover Letter field is not displayed");
        } catch (Exception e) {
            Assert.fail("Error message not displayed: " + e.getMessage());
        }
    }
}
"""

def generate_gherkin_feature(user_story, detail_level):
    if detail_level == "Detailed":
        custom_prompt_template = """Create a comprehensive Gherkin feature file based on the provided user story. Follow these instructions to produce a detailed output:
                                Instructions:
                                1. Carefully analyze the user story and extract all possible scenarios, including edge cases and alternative flows.
                                2. Create multiple scenarios to cover various aspects of the feature, considering different user inputs, conditions, and outcomes.
                                3. Use descriptive and specific language for each step.
                                4. Include background steps if there are common preconditions for all scenarios.
                                5. Utilize scenario outlines with examples for data-driven scenarios when appropriate.
                                6. Consider both happy path and negative scenarios.
                                7. Add relevant tags to group related scenarios.
                                FORMAT:
                                Feature: [Feature Name]
                                As a [type of user]
                                I want [goal]
                                So that [benefit]
                                Background: (if applicable)
                                    Given [common preconditions]

                                @tag1 @tag2
                                Scenario: [Primary Scenario Name]
                                    Given [precondition]
                                    And [another precondition]
                                    When [action]
                                    And [another action]
                                    Then [expected result]
                                    And [another expected result]

                                @tag3
                                Scenario: [Alternative Scenario Name]
                                    Given [different precondition]
                                    When [different action]
                                    Then [different expected result]

                                @tag4
                                Scenario Outline: [Data-driven Scenario Name]
                                    Given [precondition with <variable>]
                                    When [action with <variable>]
                                    Then [expected result with <variable>]

                                    Examples:
                                    | variable | other_variable |
                                    | value1   | other_value1   |
                                    | value2   | other_value2   |

                                @negative @edge_case
                                Scenario: [Negative Scenario Name]
                                    Given [precondition for negative case]
                                    When [action that should fail]
                                    Then [expected error or failure result]

                                Context: {context}
                                Answer:
                                """
    else:
            custom_prompt_template = """Create the feature file as per BDD framework for provided test case in question. Follow below instructions to produce output
                    Instructions
                    1. Carefully analyse the test case provided with each step.
                    2. Do not skip or ignore any test step as these are critical for feature file creation.
                    3. Your correct interpretation would help subsequent prompt to provide correct responses.
                    4. Strictly adhere to the format provided below.
                    FORMAT
                    Feature: [Feature Name]
                    Description: [Brief description of the functionality under test]

                    Scenario Outline: [Scenario Name]
                    Given [Preconditions]
                    When [Event or Action]
                    Then [Expected result]
                    And [Additional expected results]
                    But [Optional: Negative expected results]

                    Examples:
                    | [Parameter 1] | [Parameter 2] |... |
                    | [Value 1]     | [Value 2]     |... |
                    | [Value 3]     | [Value 4]     |... |

                    Context: {context}
                    Answer:
                    """

    prompt = custom_prompt_template.format(context=user_story)
    messages = [ChatMessage(role="user", content=prompt)]
    response = llm.chat(messages)
    return response.message.content

def streamlit_webagent_demo(objective: str, url: str):
    st.write(f"Objective: {objective}")
    st.write(f"Starting URL: {url}")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    selenium_driver = SeleniumDriver(driver=driver)
    world_model = WorldModel.from_context(context)
    action_engine = ActionEngine.from_context(context, selenium_driver)
    agent = WebAgent(world_model, action_engine)
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    # Navigate to the initial URL
    agent.get(url)
    # Run the agent
    for step in range(agent.n_steps):
        status_text.text(f"Step {step + 1}/{agent.n_steps}")
        # Run a single step
        result = agent.run(objective)
        # Update progress
        progress_bar.progress((step + 1) / agent.n_steps)
        # Display current URL
        st.write(f"Current URL: {agent.driver.get_url()}")
        # Display screenshot
        screenshot = Image.open(BytesIO(agent.driver.get_screenshot_as_png()))
        st.image(screenshot, caption=f"Step {step + 1} Screenshot", use_column_width=True)
        # Display action taken
        st.write(f"Action taken: {result.instruction}")
        # Display output
        if result.output:
            st.write(f"Output: {result.output}")
        # Check if objective is reached
        if result.success:
            st.success("Objective reached!")
            break
    # Final status
    if not result.success:
        st.error("Failed to reach the objective within the given steps.")
    # Display final result
    st.write("Final Result:")
    st.json(result.__dict__)

def identify_elements_and_generate_csv(url, output_file='elements.csv'):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)

    def highlight_element(element):
        driver.execute_script(
            "arguments[0].setAttribute('style', arguments[1]);",
            element,
            "border: 2px solid red;"
        )

    def add_id_overlays(elements):
        js_script = """
        function addIdOverlay(element, id) {
            const rect = element.getBoundingClientRect();
            const overlay = document.createElement('div');
            overlay.textContent = id;
            overlay.style.position = 'absolute';
            overlay.style.backgroundColor = 'rgba(255, 0, 0, 0.7)';
            overlay.style.color = 'white';
            overlay.style.padding = '2px 5px';
            overlay.style.borderRadius = '3px';
            overlay.style.fontSize = '12px';
            overlay.style.zIndex = '10000';
            overlay.style.pointerEvents = 'none';
            overlay.style.left = (rect.left - 25) + 'px';
            overlay.style.top = (rect.top - 25) + 'px';

            if (rect.left < 30) {
                overlay.style.left = rect.right + 'px';
            }

            if (rect.top < 30) {
                overlay.style.top = rect.bottom + 'px';
            }
            document.body.appendChild(overlay);
        }

        const elements = arguments[0];
        for (let i = 0; i < elements.length; i++) {
            addIdOverlay(elements[i], i);
        }

        """
        driver.execute_script(js_script, elements)
    try:
        # Wait for the page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        # Find all elements
        elements = driver.find_elements(By.XPATH, "//*[@id]")
        # Highlight elements and add overlays
        for element in elements:
            highlight_element(element)
        add_id_overlays(elements)
        # Prepare data for CSV
        element_data = []
        for i, element in enumerate(elements):
            element_id = element.get_attribute("id")
            element_xpath = driver.execute_script(
                "function getXPath(element) {"
                "   if (element.id !== '')"
                "       return 'id(\"' + element.id + '\")';"
                "   if (element === document.body)"
                "       return element.tagName;"
                "   var ix = 0;"
                "   var siblings = element.parentNode.childNodes;"
                "   for (var i = 0; i < siblings.length; i++) {"
                "       var sibling = siblings[i];"
                "       if (sibling === element)"
                "           return getXPath(element.parentNode) + '/' + element.tagName + '[' + (ix + 1) + ']';"
                "       if (sibling.nodeType === 1 && sibling.tagName === element.tagName)"
                "           ix++;"
                "   }"
                "}"
                "return getXPath(arguments[0]);", element
            )
            element_data.append([i, element_id, element_xpath])
        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'Element ID', 'XPath'])
            writer.writerows(element_data)
        print(f"Element data has been written to {output_file}")
        # Keep the browser open for inspection
        input("Press Enter to close the browser...")
    finally:
        driver.quit()

def setup_interactive_browser(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)
    
    js_code = """
    var selectedElements = [];
    document.body.addEventListener('click', function(event) {
        event.preventDefault();
        var element = event.target;
        var elementInfo = {
            tag: element.tagName,
            id: element.id,
            class: element.className,
            text: element.textContent.trim().substring(0, 50)
        };
        var index = selectedElements.findIndex(e => e.id === elementInfo.id && e.tag === elementInfo.tag);
        if (index > -1) {
            element.style.border = '';
            selectedElements.splice(index, 1);
        } else {
            element.style.border = '2px solid red';
            selectedElements.push(elementInfo);
        }
        localStorage.setItem('selectedElements', JSON.stringify(selectedElements));
    }, true);
    """
    driver.execute_script(js_code)
    return driver

def get_selected_elements(driver):
    try:
        elements = driver.execute_script("return localStorage.getItem('selectedElements');")
        return json.loads(elements) if elements else []
    except WebDriverException:
        return None

def generate_test_scenarios(url, selected_elements, screenshot):
    img_str = ""
    if screenshot:
        buffered = io.BytesIO(screenshot)
        img_str = base64.b64encode(buffered.getvalue()).decode()
    
    role = "You are a Software Test Consultant with expertise in web application testing"
    prompt = f"""Generate test ideas based on the selected elements of the webpage. 
    Focus on user-oriented tests that cover functionality, usability, and potential edge cases. 
    Include both positive and negative test scenarios. Consider the element types and their potential interactions.

    Page URL: {url}
    
    Selected Elements:
    {selected_elements}
    
    Screenshot: {"Available" if img_str else "Not available"}

    Please provide a mix of positive and negative test scenarios, considering the interactions between the selected elements.
    Format the output as a numbered list of test scenarios.

    Format the output as the following example:
    Positive Tests:
    - <Idea 1>
    - <Idea 2>

    Negative Tests:
    - <Idea 1>
    - <Idea 2>

    Edge Cases and Usability Tests:
    - <Idea 1>
    - <Idea 2>
    """
    messages = [ChatMessage(role="system", content=role),
                ChatMessage(role="user", content=prompt)]
    response = mm_llm.chat(messages)
    return response.message.content

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Streamlit interface
def streamlit_interface():
    st.set_page_config(page_title="SDET-Genie", page_icon="🧞", layout="wide")
    
    # Custom CSS with animations
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background: linear-gradient(-45deg, #4169E1, #800080, #FF69B4, #1E90FF);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: #FFFFFF;
    }
    .selectbox-wrapper {
        background: linear-gradient(-45deg, #4169E1, #800080, #FF69B4, #1E90FF);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: #FFFFFF;
    }
    .title {
        font-family: 'Pacifico', cursive;
        font-size: 4rem;
        text-align: center;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 0;
    }
    .subtitle {
        font-family: 'Arial', sans-serif;
        font-size: 1.5rem;
        text-align: center;
        color: #FFFFFF;
        margin-top: 0;
    }
    .stButton > button {
        color: #FFFFFF;
        background-color: #800080;
        border: 2px solid #4169E1;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #4169E1;
        transform: scale(1.05);
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: rgba(255, 105, 180, 0.1);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        background-color: rgba(255, 105, 180, 0.3);
        transform: scale(1.02);
    }
    .stRadio > div {
        background-color: rgba(255, 105, 180, 0.1);
        padding: 10px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stRadio > div:hover {
        background-color: rgba(255, 105, 180, 0.3);
        transform: scale(1.02);
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title fade-in'>🧞‍♀️🧞‍♂️ SDET-Genie 🧞‍♀️🧞‍♂️</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subtitle fade-in'>All your QA Needs in One Tool</h3>", unsafe_allow_html=True)

    # Sidebar with animation
    lottie_coding = load_lottieurl('https://lottie.host/5a6d06b1-45f0-4b0e-8ac3-271f7791d9f8/bdKsVI11bC.json')
    with st.sidebar:
        st_lottie(lottie_coding, speed=1, height=200, key="coding")

    option = st.sidebar.selectbox("Select an option", 
                                  ["Generate Test Scenarios",
                                   "Identify Page Elements",
                                   "Generate BBD Gherkin Feature Steps", 
                                   "Generate Automation Scripts", 
                                   "Web Agent scraping"
                                   ])
    # Main content area
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if option == "Generate Automation Scripts":
            lottie_robot = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_M9p23l.json')
            st_lottie(lottie_robot, speed=1, height=300, key="robot")
            st.title("Generating QA Automation Scripts With Just Gherkin Steps")
            st.write("Enter a URL, Gherkin feature steps, and select a language to generate automated test code.")
            url = st.text_input("URL")
            feature_content = st.text_area("Gherkin Feature Steps")
            language = st.radio("Language", ["Python", "Java"])

            if st.button("Generate Code"):
                output_path, code = main(url, feature_content, language)
                st.success(f"{language.capitalize()} test code generated at: {output_path}")
                st.code(code, language=language.lower())

        elif option == "Generate BBD Gherkin Feature Steps":
            lottie_steps = load_lottieurl('https://lottie.host/cacd1d54-83e0-40dd-bfe6-fc71b136d6ee/kZjrOTkQdO.json')
            st_lottie(lottie_steps, speed=1, height=300, key="steps")
            st.title("Generate BDD Gherkin Feature Steps")
            st.write("Enter a user story to generate Gherkin feature steps.")
            user_story = st.text_area("User Story")
            detail_level = st.radio("Choose detail level", ["Simple", "Detailed"])
            if st.button("Generate Gherkin Feature"):
                gherkin_feature = generate_gherkin_feature(user_story, detail_level)
                st.success("Gherkin Feature Generated")
                st.code(gherkin_feature, language="gherkin")

        elif option == "Web Agent scraping":
            lottie_web = load_lottieurl('https://lottie.host/78d638e9-e95a-42b8-944f-e65b95120010/sl2gbZWLFk.json')
            st_lottie(lottie_web, speed=1, height=300, key="web")
            st.title("Web Agent Demo")
            st.write("Enter an objective and URL to start the Web Agent demo.")
            objective = st.text_input("Objective")
            url = st.text_input("Starting URL")

            if st.button("Start Demo"):
                streamlit_webagent_demo(objective, url)

        elif option == "Identify Page Elements":
            lottie_search = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json')
            st_lottie(lottie_search, speed=1, height=300, key="search")
            st.title("Identify Page Elements")
            st.write("Enter a URL to identify all elements with IDs and generate a CSV file.")
            url = st.text_input("URL")
            output_file = st.text_input("Output CSV file name", value="elements.csv")
            if st.button("Identify Elements"):
                with st.spinner("Identifying elements and generating CSV..."):
                    identify_elements_and_generate_csv(url, output_file)
                st.success(f"Element data has been written to {output_file}")
                # Display the CSV content
                with open(output_file, 'r') as csvfile:
                    csv_content = csvfile.read()
                st.download_button(
                    label="Download CSV",
                    data=csv_content,
                    file_name=output_file,
                    mime="text/csv"
                )
                st.code(csv_content, language="csv")

        elif option == "Generate Test Scenarios":
            lottie_test = load_lottieurl('https://lottie.host/8bf5d28b-0256-41ef-9719-b2c1a7369150/48W8wvE9Gd.json')
            st_lottie(lottie_test, speed=1, height=300, key="test")
            st.title("Interactive Test Scenario Generator")
            url = st.text_input("Enter the URL of the webpage you want to test:")
            
            if 'driver' not in st.session_state:
                st.session_state.driver = None
            
            if url and st.button("Start Element Selection"):
                if st.session_state.driver:
                    try:
                        st.session_state.driver.quit()
                    except:
                        pass
                
                st.session_state.driver = setup_interactive_browser(url)
                st.write("Browser opened. Please select elements on the webpage.")
                st.write("Click 'Generate Test Scenarios' when you're done selecting elements.")
            
            if st.session_state.driver:
                if st.button("Check Selected Elements"):
                    selected_elements = get_selected_elements(st.session_state.driver)
                    if selected_elements:
                        st.write("Currently selected elements:")
                        st.write(selected_elements)
                    else:
                        st.write("No elements selected yet.")
                
                if st.button("Generate Test Scenarios"):
                    selected_elements = get_selected_elements(st.session_state.driver)
                    if selected_elements is not None:
                        try:
                            screenshot = st.session_state.driver.get_screenshot_as_png()
                        except WebDriverException:
                            st.error("Unable to capture screenshot. Browser may have been closed.")
                            screenshot = None
                        
                        if screenshot:
                            st.image(Image.open(io.BytesIO(screenshot)), caption="Webpage with Selected Elements", use_column_width=True)
                        
                        test_scenarios = generate_test_scenarios(url, selected_elements, screenshot)
                        st.write("Generated Test Scenarios:")
                        st.write(test_scenarios)
                    else:
                        st.error("Unable to retrieve selected elements. Please restart the element selection process.")
                    
                    # Close the browser after generating scenarios
                    try:
                        st.session_state.driver.quit()
                    except:
                        pass
                    st.session_state.driver = None
 
if __name__ == "__main__":
    streamlit_interface()
