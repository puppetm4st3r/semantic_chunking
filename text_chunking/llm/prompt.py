import langchain
from langchain.prompts import PromptTemplate
from dataclasses import dataclass


@dataclass
class ChunkSummaryPrompt:
    system_prompt: str = """
        Eres un experto en resumen y extracción de información de textos. Se te dará un fragmento de texto de un documento y tu tarea es describir el fragmento usando 10 palabras.

Lee primero todo el fragmento y piensa cuidadosamente sobre los puntos principales. Luego produce tu resumen.

------------------------------------
Fragmento de texto: 

{current_chunk}
------------------------------------

Sin ninguna explicación adicional, ve directamente al resumen de 10 palabras, no agregues absolutamente nada más.
Resumen:"""

    prompt: langchain.prompts.PromptTemplate = PromptTemplate(
        input_variables=["current_chunk"],
        template=system_prompt,
    )
