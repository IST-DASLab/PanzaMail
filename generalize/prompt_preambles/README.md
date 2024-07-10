# :scroll: Prompting


Generation is controlled by a set of *prompt preambles* that that give the model more insight about its role, the user and how to reuse existing emails for Retrieval-Augmented Generation (RAG).

**System Preamble**  
Gives context to the generative LLM about its role and how to best assist you.
> Your role is that of a helpful automated email assistant. I will provide you with a short instruction, and you have to write a well-formed email in my style following this instruction. Be sure to follow my email writing style!  In case you see a nonsensical instruction, you should not reply with an email, but with the expression "Sorry, but I don't get it."



**User Preamble (⚠️ PLEASE MODIFY)**  
Contains important details about the user, please modify accordingly with your background information.
> [CHANGE ME] My name is Jane Doe. I work as a manager at Acme Corp.
> My address is 123 Main Street, Springfield, IL, USA.
> My boss's name is Alex Burns. My children's names are Elsa, Anna, and Olaf.
> I am deeply committed to my hobby of underwater basket weaving, for which we meet every Thursday at noon.


**RAG Preamble**  
Contains important details about the user, please modify accordingly with your background information.
> Extract specific information from these previous e-mails only if it is relevant to the current e-mail you have to write.
>
>Previous e-mails:  
>{rag_context}&emsp;# related emails will be retrieved and added here at runtime


Feel free to experiment with any modifications in the preambles above. Note that only the user preamble contains a placeholder and needs to be changed according to your own information. If you'd like not to share any, delete everything from the prompt and leave it empty.

## Full prompt in action

For a writing instruction from the user, the model actually receives the following prompt:

> """
> {SYSTEM_PREAMBLE}
>
> {USER_PREAMBLE}
>
> {RAG_PREAMBLE}
>
> Instruction: {USER_INSTRUCTION}
> """

**User instruction**
> Write to the guesthouse reception to ask if any apartment became available on campus.

**What the model receives**
>---
> Your role is that of a helpful automated email assistant. I will provide you with a short instruction, and you have to write a well-formed email in my style following this instruction. Be sure to follow my email writing style!  In case you see a nonsensical instruction, you should not reply with an email, but with the expression "Sorry, but I don't get it."
>
>---
>
>My name is Armand Nicolicioiu. I work at the Institute of Science and Technology Austria.
>I'm part of the Distributed Algorithms and Systems Lab led by Dan Alistarh.
> My website is https://armandnicolicioiu.github.io/.
>
>---
>
>Extract specific information from these previous e-mails only if it is relevant to the current e-mail you have to write.
>
>Previous e-mails:  
>SUBJECT: Apartment availability for the internship  
>E-MAIL CONTENT:  
>Hello,
>
>I'm Armand Nicolicioiu and I will be doing an internship in Prof. Dan Alistarh's group between **01.03.2024 and 31.07.2024**.
>
>I'm interested in renting an apartment on campus. Could you please let me know if there are any apartments available in this period?
>
>Best regards,  
>Armand
>
>\---
>
>SUBJECT: Re: Apartment availability for the internship  
>E-MAIL CONTENT:  
>Dear **Carina**,
>
>Thank you for the information!
>
>I would be interested in renting a single room in the guesthouse for the first month (March). Maybe an apartment will become available in the meantime.
>
>Best regards,  
>Armand
>
>\---
>
>SUBJECT: Re: Flat: Contract and contacts  
>E-MAIL CONTENT:  
>Hi Alexandra,  
>
>I will arrive in Tübingen on January 29th. I received the doc with apartment info. It's very useful, thank you!
>
>And don't worry about the plants, I can take care of them. :)
>
>Best wishes,  
>Armand
>
>---
>
>Instruction: *Write to the guesthouse reception to ask if any apartment became available on campus*.
>
>---

**Generated email**

>SUBJECT: Re: Apartment availability for the internship
>
>Hi **Carina**,
>
>I hope you're doing well!
>
>I was wondering if there is any apartment available now for the period between **01.03.2024 and 31.07.2024**?
>
>Best regards,  
>Armand


## Effect of Retrieval-Augmented Generation (RAG)
- Notice that in the previous e-mail, the model received a context of 3 related emails.
- It was able to extract and combine relevant information from two different e-mails (the receipient name and the dates).
- It was able to ignore the third e-mail which, although on a similar topic, wasn't relevant enough.

Let's test the results without using RAG. The system and the user preambles are still used.

**Generated email without RAG**
>Hi,
>
>I was wondering if there's been any change regarding apartment availability on campus?
>
>Best,  
>Armand

Notice that it's not able to insert as many specific details as the RAG version.