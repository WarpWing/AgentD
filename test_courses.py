import os
import nextcord
from nextcord.ext import commands
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

def setup_driver():
    chromedriver_path = os.path.join(os.getcwd(), 'chromedriver.exe')
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(chromedriver_path)
    return webdriver.Chrome(service=service, options=options)

def get_courses(subject):
    driver = setup_driver()
    driver.get("https://cliq.dickinson.edu/apps/registrar/course_criteria_public.cfm")
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f"//a[contains(text(), '{subject}')]"))
        )
        subject_link = driver.find_element(By.XPATH, f"//a[contains(text(), '{subject}')]")
        subject_link.click()
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "table-striped"))
        )
        courses = []
        rows = driver.find_elements(By.XPATH, "//table[@class='table table-striped ']/tbody/tr")
        for row in rows[1:]:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 9:
                course = {
                    "CRN": cols[0].text,
                    "Subject": cols[1].text,
                    "Number": cols[2].text,
                    "Section": cols[3].text,
                    "Title": cols[4].text.split('\n')[0],
                    "Schedule": cols[4].text.split('\n')[1] if len(cols[4].text.split('\n')) > 1 else "",
                    "Cross-listed": cols[6].text,
                    "Capacity": cols[7].text,
                    "Requested/Registered": cols[8].text
                }
                courses.append(course)
    finally:
        driver.quit()
    return courses

bot = commands.Bot()

@bot.slash_command(description="Retrieve courses for a given subject")
async def get_courses_command(interaction: nextcord.Interaction, subject: str):
    await interaction.response.defer()
    subject = subject.upper()
    courses = get_courses(subject)
    
    if not courses:
        await interaction.followup.send(f"No courses found for subject: {subject}")
        return

    embeds = []
    for i in range(0, len(courses), 25):  # Discord has a limit of 25 fields per embed
        embed = nextcord.Embed(title=f"Courses for {subject}", color=0x00ff00)
        for course in courses[i:i+25]:
            value = (
                f"Number: {course['Number']}\n"
                f"Section: {course['Section']}\n"
                f"Schedule: {course['Schedule']}\n"
                f"Cross-listed: {course['Cross-listed']}\n"
                f"Capacity: {course['Capacity']}\n"
                f"Requested/Registered: {course['Requested/Registered']}"
            )
            embed.add_field(name=f"{course['CRN']} - {course['Title']}", value=value, inline=False)
        embeds.append(embed)

    await interaction.followup.send(embeds=embeds)

bot.run("Banana Bread :D")