# First, you need to install the reportlab library. 
# You can do this by running the following command in your terminal or command prompt:
# pip install reportlab

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.units import inch
import os

# --- Story 1: A Magical Quest ---
story1_title = "The Whispering Compass and the 3 Lost Stars"
story1_text = """
Once, in a village nestled between two towering mountains, lived a young girl named Elara who was the apprentice to the town's elderly mapmaker, a man named Cassian with eyes that held the wisdom of 150 years. Elara didn’t just draw maps; she charted the skies, her heart filled with a wonder for the cosmos. One evening, a great sadness fell as 3 of the most brilliant stars in their brightest constellation, The Silver Swan, vanished, leaving gaping holes of darkness. The oldest legends whispered that without the complete Swan, the magical rivers that watered their crops would cease to flow. A formal council of elders was held, and the 7 members agreed the situation was dire. The air in the council chamber grew heavy with unspoken fear. Cassian gave Elara a special compass, a Whispering Compass with a needle of pure moonlight, said to point the way to anything lost. "The journey will be perilous," he warned, "but your heart is true." Elara knew she had to embark on this adventure for the 150 people who called her village home. She packed a bag with a blanket, bread, and the mystical compass, feeling the immense weight of her world on her small shoulders.

Her first destination, as whispered by the compass, was the Sunken Grove, a forest where the trees grew upside down. It was said that 25 mischievous sprites guarded the Grove, delighting in leading travelers astray with riddles. Before she left, Cassian had advised her, "Remember that charm is often more powerful than force." Recalling his words, Elara cleverly brought along a pouch containing 12 glistening river pearls and a jar of sweet honey. Upon meeting the sprites, she offered them her gifts, and their tiny faces lit up with joy. In their gratitude, they created a glowing path of fireflies to guide her. The head sprite, with wings like stained glass, told her the first star had been caught in the branches of the oldest tree, a giant weeping willow whose tears were said to be morning dew. Elara carefully climbed the ancient tree and retrieved the first star, which felt warm and pulsed with a gentle light in her hands. She thanked the sprites, who promised to keep the path lit for her return.

With the first star in a velvet pouch, the compass needle spun wildly before pointing toward the Crystal Caves behind the great waterfall. The journey took 4 long days, and she had to cross the Chattering River, which was home to exactly 11 singing fish. This was a place of known danger, a chattering torrent of water that had claimed unwary travelers before. The fish, however, were not malicious, and they gave her advice on the safest places to cross in exchange for a story about the stars. She spent hours studying a puzzle lock at the cave's entrance, her mind racing back to the thousands of star charts she had studied. Finally, she recognized the pattern of the long-lost "Silent Dragon" and carefully arranged the crystals to match it. Inside, the second star was on a giant quartz formation, its light reflecting in a million directions. Retrieving it was simple, but the journey had tested her mind just as the Grove had tested her kindness. She knew the final test would be the most difficult.

The compass now pointed straight up, towards the highest peak of the two mountains, a place known as the Dragon's Tooth. The final star, the compass whispered, had been taken by a young griffin who was lonely and wanted a friend that sparkled. Elara began the arduous climb. The air grew thin and cold, and a silence so profound settled around her that she could hear her own heartbeat. When she reached the peak, she found the young griffin in a huge nest, nudging the star with its beak. Instead of trying to snatch the star, Elara sat down and began to talk to the griffin, telling it stories. She spent the entire day with the creature, sharing her last piece of bread. The griffin, who had never had a visitor, was so moved by her gentleness that it nudged the star back into her hands. Elara promised she would visit again, and with all 3 stars in her pouch, she made her way back down the mountain.

The return to the village was met with a celebration that lasted for a week, a feast grander than any had seen. Elara, guided by Cassian, climbed to the top of the tallest bell tower. One by one, she released the 3 recovered stars. They flew upwards, settling back into their rightful places, and the Silver Swan shone more brilliantly than ever. The magical rivers began to flow with renewed strength, promising a bountiful harvest for all. Elara was no longer just an apprentice; she was a hero, the brave Starchaser. She continued to make her maps, but now, she added new constellations of her own invention: one of a kind girl, one of a lonely griffin, and one of a Whispering Compass, forever immortalizing the adventure. And sometimes, on clear nights, she would wave up at the young griffin, who would fly past the moon, dipping a wing in greeting.
"""


# --- Story 2: A Whimsical Animal Tale ---
story2_title = "Barnaby the Badger's Extraordinary Emporium"
story2_text = """
Deep in the heart of the Whispering Woods, there was a most unusual shop run by a badger named Barnaby. His Emporium was no ordinary store; it catered to the peculiar needs of the woodland creatures. For instance, Barnaby stocked 12 different kinds of moss for discerning frogs. His shop was a bustling hub of commerce, with a bell that jingled merrily. This particular autumn, a panic was rippling through the community of 45 squirrel families because the acorn harvest had been poor. For weeks, they had tried to gather nuts from the 5 best oak trees, but were thwarted each time. A silence so profound had fallen over the squirrel village, born of worry for the coming winter. The squirrel council, a group of 7 elder squirrels, was at a loss. The air in their hollow log grew heavy with unspoken fear.

The problem was a flock of 25 very grumpy blue jays who had claimed the best oak trees, chasing away any squirrel who dared venture near. Mr. Fitzwilliam, the head of the squirrel council, scurried into Barnaby's Emporium, his tail twitching with anxiety. "Barnaby, old friend, we are in a terrible predicament! Winter is coming and our stores are empty!" Barnaby listened patiently, stroking his chin. Before devising a plan, he decided to consult the wisest creature he knew: a tortoise named Solomon, who was rumored to be over 150 years old and lived by Miller's Pond. "The jays are vain and easily distracted by shiny things," Solomon advised slowly. "A grand illusion is needed, not a direct fight. Charm is often more powerful than force." With this wisdom, Barnaby knew exactly what to do.

Returning to his shop, Barnaby scurried to his inventions section. He gathered his materials: 10 feet of stretchy vine, 2 large maple leaves, and the crucial component, a pot of Sunbeam Sap, known to glow with a gentle light. His plan was to create a distraction so magnificent the jays would forget the acorns. He worked for 4 days, his paws a blur, with the help of 3 industrious field mice. They crafted "The Shimmerwing," a device like a giant, fantastical butterfly. The maple leaves formed the wings, held by a willow frame and stitched with spider silk. He then coated the surface with the glowing sap, which he had to purchase from a grumpy gnome for 3 shiny pebbles. Mr. Fitzwilliam and the squirrel council watched in awe as Barnaby unveiled his creation. Hope filled the squirrels for the first time in weeks.

The following morning, the squirrels put the plan into action. Under the cover of dawn, 2 brave squirrels launched the Shimmerwing into the sky. It soared and dipped, its glowing wings creating shifting patterns of light. Just as Solomon and Barnaby predicted, the 25 blue jays were utterly mesmerized. They stopped their squawking and flew up to chase the beautiful, dancing light. While they were distracted, the 45 squirrel families swarmed the 5 oak trees, working with incredible speed. In less than an hour, they had gathered enough acorns to last the entire winter. They scurried back to their homes, their winter stores now filled to the brim, chattering with joy and relief. The grandest celebration followed.

The blue jays chased the Shimmerwing all the way to Miller's Pond, where the vine snagged on a cattail, and the device floated down, its light fading. By the time the jays flew back, the acorns—and the squirrels—were long gone. That evening, the entire squirrel community held a grand feast in Barnaby's honor, presenting him with the single largest acorn they found, which he placed in a velvet box. Barnaby's Emporium had once again saved the day, not with weapons, but with a clever idea. The story of the Shimmerwing became a local legend, a tale told to young woodland creatures about how ingenuity and seeking wisdom were far more powerful than being the biggest or the loudest. And Barnaby the badger just smiled, polished his spectacles, and waited for the little bell on his door to jingle.
"""


# --- Story 3: A Creative and Thoughtful Tale ---
story3_title = "The Girl Who Painted the Silence"
story3_text = """
In a faraway city made of pale grey stone, where even the sky was ash, lived a girl named Lily. In this city, there was no music or laughter; everything was muted and orderly, just as the city's 5 somber-faced rulers liked it. For over 150 years, since the "Great Silencing," the people had forgotten what it was like to sing or to see a field of red poppies. Lily, however, was different. She often visited the city's only library, where a kind, elderly librarian named Elara (a descendant of the Starchaser) secretly let her see a forbidden book. The book had just one faded picture of a yellow dandelion, but it was enough to plant a seed of rebellion in Lily's heart. Lily had a secret: 7 small, enchanted paintbrushes, a gift from her grandmother. Her grandmother had told her, "These don't just paint with color; they paint with life itself."

One night, Lily snuck out with her 7 brushes. She went to the great, grey town square, dominated by a silent stone fountain that hadn't seen water in decades. The silence was so profound, it felt like a heavy blanket smothering the world. With a deep breath, she chose the first brush, the one with a sky-blue handle. She touched its tip to the dry stone. The moment it made contact, a soft, magical sound began to fill the air—the gentle burble of a flowing stream. The blue color spread like water, filling the fountain with the illusion of sparkling water, its splashing echoing through the square. This was a sound no one under the age of 150 had ever heard. People began to peek out of their windows, their eyes wide with disbelief.

Emboldened, Lily picked up her green and brown brushes. She painted the cold flagstones, and with every stroke, the sound of rustling leaves and birdsong joined the melody. Lush, green grass seemed to sprout from the stones. The towering grey buildings began to look like ancient trees. Then she grabbed her fourth brush, with a fiery red handle, and painted a long path. As she painted, it bloomed with thousands of imaginary red poppies, and a new sound was added: the happy hum of busy bumblebees. The city was transforming, the silence being replaced by a chorus of life. She was breaking countless rules, but a new feeling—hope—was taking root in the city's heart.

The 5 city rulers, disturbed by the chaos, marched into the square. "Cease this at once! You are breaking rule number 12, section B: No unauthorized sounds! And rule number 25: No colors brighter than pavement grey!" the head ruler boomed. But Lily was no longer afraid. She looked at the faces of her neighbors, seeing smiles she had never seen before. She took her final 3 brushes—yellow, orange, and purple—and with a great, sweeping motion, she painted the sky. A magnificent, painted sunrise spread across the grey canvas above, and a grand, orchestral symphony filled the air. The music was so beautiful the rulers stopped, stunned into silence. Even their own guards, 45 of them in all, lowered their staves, mesmerized by the sight.

The people of the city slowly began to step into the square. An old woman began to hum along, and then a young boy started to laugh, a sound that was foreign yet wonderful. Soon, the entire city was filled with laughter and singing, a joyful noise that overpowered the rulers' demands. The colors Lily painted seemed to seep into the very stones of the city, and they stayed, a permanent reminder. The council of rulers, seeing they had lost their hold on the city's grey heart, had no choice but to repeal their silent rules. Lily and her 7 paintbrushes had not just painted a picture; she had painted a new beginning, proving that it only takes one person with courage and a splash of color to paint over the deepest silence.
"""

def create_fairy_tale_pdf(title, content, filename):
    """
    Generates a PDF for a single fairy tale.
    """
    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Overwriting.")
        # os.remove(filename) # Optional: remove if you don't want to overwrite

    doc = SimpleDocTemplate(filename)
    
    # Define a custom style for the story text
    styles = getSampleStyleSheet()
    kid_style = ParagraphStyle(
        name='KidStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=15,
        leading=22, 
        alignment=TA_JUSTIFY,
    )
    
    # Add title style
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['h1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        alignment=1 # Center alignment
    )

    flowables = []
    
    flowables.append(Paragraph(title, title_style))
    flowables.append(Spacer(1, 0.4 * inch))
    
    paragraphs = content.strip().split('\n\n')
    
    for para_text in paragraphs:
        p = Paragraph(para_text.strip(), kid_style)
        flowables.append(p)
        flowables.append(Spacer(1, 0.2 * inch)) 
        
    try:
        doc.build(flowables)
        print(f"Successfully created '{filename}'")
    except Exception as e:
        print(f"Error creating '{filename}': {e}")

def main():
    """
    Main function to generate all fairy tale PDFs.
    """
    stories_data = {
        story1_title: story1_text,
        story2_title: story2_text,
        story3_title: story3_text,
    }

    print("Starting to generate fairy tale PDFs...")

    for title, content in stories_data.items():
        # Create a more robust, clean filename from the title
        filename = "_".join(title.split(' ')[:3]).replace(',', '').replace('.', '') + ".pdf"
        create_fairy_tale_pdf(title, content, filename)
        
    print("\nAll PDFs have been generated successfully!")
    print(f"Check the directory '{os.getcwd()}' for your files.")

if __name__ == '__main__':
    main()
