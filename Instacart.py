from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time

profile_path = r"C:\Users\Alexander\AppData\Local\Google\Chrome\User Data\Default"

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
# chrome_options.add_argument("user-data-dir=" + profile_path)
# chrome_options.add_argument("--start-maximized")


# Initialize the Chrome WebDriver
driver = webdriver.Chrome(options=chrome_options)

# URL of the Instacart CVS storefront
login_url = "https://www.instacart.com/login"
store_url = "https://www.instacart.com/store/cvs"


# Open the URL

driver.get(login_url)
driver.implicitly_wait(8)
email_input = driver.find_element(By.CSS_SELECTOR, "input[type='email']")
email_input.send_keys("halperna22@gmail.com")
driver.implicitly_wait(5)
email_input = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
email_input.send_keys("Treehacks1!")
submit_button = driver.find_element(
    By.CSS_SELECTOR, "button[class='e-ztomkz']")
submit_button.click()
time.sleep(5)
# CVS_button = driver.find_element(By.CSS_SELECTOR, "img[alt='CVS®']")
# CVS_button.click()
# driver.implicitly_wait(5)
# span_element = driver.find_element(By.XPATH, "//span[text()='Medicine']")
# CVS_button.click()
product_link = "https://www.instacart.com/products/84345-sambucol-cold-flu-relief-black-elderberry-quick-dissolve-tablets-30-ea?retailerSlug=cvs"
driver.get(product_link)

time.sleep(5)

add_button = driver.find_element(
    By.CSS_SELECTOR, "button[class='e-1mchykm']")

add_button.click()

driver.implicitly_wait(20)
driver.get("https://www.instacart.com/store/checkout_v4?sid=53040")

# text_area = driver.find_element(By.ID, "deliveryInstructions")
# text_area.send_keys("Please leave at the front door")
# save_and_continue = driver.find_element(By.CLASS_NAME, "e-15utg5h")
# save_and_continue.click()
# time.sleep(8)
choose_delivery_time = driver.find_element(By.CLASS_NAME, "e-rloafg")
choose_delivery_time.click()
# phone_number = driver.find_element(By.CSS_SELECTOR, "input[type='tel']")
# phone_number.send_keys("2034518641")
# submit_phone_number = driver.find_element(By.CLASS_NAME, "e-sp84se")
# submit_phone_number.click()
# time.sleep(4)
final_continue = driver.find_element(By.CLASS_NAME, "e-15utg5h")
final_continue.click()
time.sleep(8)
# Find the span element by its text content "Place order"
place_order_span = driver.find_element(
    By.XPATH, "//span[text()='Place order']")

# Click the span element
place_order_span.click()
# cough_drop_button = driver.find_element(
#     By.CSS_SELECTOR, "button[aria-label='Add 1 item Halls Relief Honey Lemon Cough Suppressant/Oral Anesthetic Menthol Drops']")
# cough_drop_button.click()

# driver.implicitly_wait(8)
# driver.get(store_url)


# You can add additional code here to interact with the page as needed

# When you're done, you can close the browser
# driver.quit()
