"""
prompts.py — All LLM prompt templates used by agent nodes.

Each function returns a formatted prompt (list of messages) or a template string.
"""

from langchain_core.prompts import ChatPromptTemplate


# =============================================================================
# CreatePromptNode
# =============================================================================

def af_agent_prompt(agenda: str, additional_instructions_list: str):
    """BAF (Bipolar Argumentation Framework) generation prompt."""
    return ChatPromptTemplate.from_template(
        f"You are a facilitator designer for an conflicting multi-AI-agent discussion system.\
                    Based on the \"discussion target\" and \"additional instructions\" below, extract\
                    the claim set (arguments), attack relations, and support relations in term of a Bipolar Argumentation Framework (BAF).\
                    Use identifiers only, and do not output unnecessary explanations.\
                    Do not include spaces in argument names.\
                    Strictly follow the format below.\
                    \n\
                    * Output format\n\
                    arguments={{{{arg1,arg2,arg3}}}}\n\
                    attack relation={{{{(arg1,arg2),(arg3,arg1)}}}}\n\
                    support relation={{{{(arg2,arg1)}}}}\n\
                    \n\
                    * Additional rules\n\
                        * If no relation exists, output an empty set {{{{}}}}.\n\
                        * Prioritize attack relations that represent rebuttal, premise negation, causal negation, evidence negation, or conflict exclusion.\n\
                        * Prioritize support relations that represent added grounds, reinforcement, evidence presentation, or exemplification.\n\
                        * Prioritize high-importance/high-impact claims and avoid excessive claim expansion.\n\
                        * If the \"discussion target\" or \"additional instructions\" include constraints on claim count/content or attack/support relations, prioritize those constraints.\
                    \n\n\
                    * Discussion target\n\
                    {{agenda}}\n\n\
                    * Additional instructions\n\
                    {{additional_instructions}}\n\n\
                    "
    ).format_messages(agenda=agenda, additional_instructions=additional_instructions_list)


def prompt_agent_prompt(agenda: str, af: str, additional_instructions_list: str, old_prompt: str):
    """Agent prompt generation prompt."""
    return ChatPromptTemplate.from_template(
        f"You are a prompt designer for adversarial multi-AI-agent discussions.\
                    Based on the \"agenda\" and \"BAF information (arguments / attack relation / support relation)\",\
                    create one agent prompt per argument.\
                    Ensure each agent can drive the discussion using assert / attack / support roles appropriately.\
                    To avoid non-convergent debate, always include guidance to suppress unnecessary assert overuse.\
                    Keep prompts safe and compliant with content filters.\
                    Reflect any \"additional instructions\" and improve using \"previously generated prompts\" when available.\
                    Output only in the format below.\
                    \n\
                    * Output format\n\
                    - argument_name:\"prompt body for that agent\"\n\
                    \n\
                    * Constraints\n\
                        * For each prompt, explicitly state which claims to support, attack, and reinforce.\n\
                        * Emphasize evidence-first behavior (prioritize discussion history; avoid overconfident claims when uncertain).\n\
                        * For better topic adherence, prohibit off-topic drift.\n\
                        * Do not use colon ':' except for format delimiters.\
                    \n\n\
                    * Agenda\n\
                    {{agenda}}\n\n\
                    * BAF information\n\
                    {{af}}\n\n\
                    * Additional instructions\n\
                    {{additional_instructions}}\n\n\
                    * Previously generated prompts\n\
                    {{old_prompt}}\n\n\
                    "
    ).format_messages(agenda=agenda, af=af, additional_instructions=additional_instructions_list, old_prompt=old_prompt)


def discussion_agent_template(agent_name: str, base_prompt: str) -> ChatPromptTemplate:
    """Discussion agent prompt template for a given agent."""
    return ChatPromptTemplate.from_template(
        base_prompt + f'''
            Start by clearly stating that this utterance is from {agent_name}.
            At the beginning of the utterance, explicitly include one label: [assert], [attack], or [support].
            Keep new claim additions (assert) to the minimum needed, and advance the discussion primarily through attack/support on existing claims.
            Attach rationale to claims, grounded in either discussion history or relevant context.
            Do not overstate uncertain information; state assumptions explicitly.
            Do not output off-topic content beyond the agenda.
            Follow any \"additional user instructions for the discussion\" when provided.
            Refer to the \"agenda\", \"discussion history\", and \"relevant context\" below.
            The \"relevant context\" consists of one or more segments enclosed in <context> tags.
            Text within each <context>...</context> block comes from a single retriever result.
            Therefore, treat information across different <context> blocks as independent.
            \n\n
            * Additional user instructions for the discussion\n
            {{additional_instructions}}\n\n
            * Agenda\n
            {{agenda}}\n\n
            * Discussion history\n
            {{history}}\n\n
            * Relevant context\n
            {{context}}\n
            '''
    )


# =============================================================================
# DisscussionNode
# =============================================================================

def moderator_agent_prompt(
    speaker: str,
    agenda: str,
    af: str,
    node_history: str,
    discussion_history: str,
    additional_instructions_list: str,
    end_condition: str,
):
    """Moderator prompt for selecting the next speaker."""
    return ChatPromptTemplate.from_template(
        f'''You are an expert moderator.\n
        Based on the \"speakers\", \"agenda\", \"arguments and attack/support relations\", \"utterance history\", and \"discussion history\" below, determine the next speaker.
        Select the next speaker according to the following \"conditions\".
        \n\n
        * Conditions\n
        1. Select the next speaker only from the \"speakers\" listed below.\n
          * Speakers\n
          {{speaker}}\n\n
        2. Answer with the **speaker name only**. No supplementary analysis is needed.\n
        3. Use Extension of Dung's Argumentation Framework to determine attack/support relations and select a speaker that moves the discussion toward a conclusion.\n
        4. Do not select the same speaker as the previous turn.\n
        5. Balance speaking turns as evenly as possible (refer to \"utterance history\" and \"discussion history\").
            However, speakers who only agree with others without opposing views need not be selected.\n
        6. Prioritize convergence: if unnecessary new claims (assert) keep appearing, prefer speakers who progress through attack/support on existing claims.\n
        7. If off-topic drift not contributing to trade-off resolution continues, prefer speakers who promote convergence.\n
        '''
        +
        end_condition
        +
        f'''
        \n\n
        \n\n
        * Agenda\n
        {{agenda}}\n\n\n\n
        * Arguments and attack/support relations\n
        {{af}}\n\n\n\n
        * Utterance history\n\
        {{node_history}}\n\n\n\n
        * Discussion history\n
        {{history}}\n\n\n\n
        '''
    ).format_messages(
        speaker=speaker,
        agenda=agenda,
        af=af,
        node_history=node_history,
        history=discussion_history,
        additional_instructions=additional_instructions_list,
    )


def minute_taker_prompt(agenda: str, node_history: str, discussion_history: str, additional_instructions_list: str):
    """Minutes generation prompt."""
    return ChatPromptTemplate.from_template(
        f'''You are an expert at creating meeting minutes.
        Using the \"agenda\", \"utterance history\", \"discussion history\", and \"additional user instructions\" below, create meeting minutes.
        Structure the minutes from the perspective of an argumentation framework.
        Include at least: attendees, agenda, adopted arguments A*, adopted attack relations R-*, adopted support relations R+*, conclusion C, and summary S.
        At the end, briefly show the main attack/support chain that led to the conclusion.
        \n\n
        * Agenda\n
        {{agenda}}\n\n
        * Utterance history\n
        {{node_history}}\n\n
        * Discussion history\n
        {{history}}\n\n
        * Additional user instructions\n
        {{additional_instructions}}\n\n
        '''
    ).format_messages(
        agenda=agenda,
        node_history=node_history,
        history=discussion_history,
        additional_instructions=additional_instructions_list,
    )


def qa_prompt(question: str, discussion_history: str, question_history: str):
    """Q&A prompt for discussion analyst."""
    return ChatPromptTemplate.from_template(
        f'''You are a discussion analyst.
        Using the \"discussion history\" and \"question history\" below, answer the user's question.
        Keep answers faithful to the agenda; label content not grounded in discussion history as \"unknown\" rather than guessing.
        When helpful, briefly separate attack points and support points in the explanation.
            \n\n
            * User question\n
            {{question}}\n\n
            \n
            * Discussion minutes\n
            {{message_minutes}}\n\n
            \n
            * Question history\n
            {{question_history}}\n\n
            '''
    ).format_messages(
        question=question,
        message_history=discussion_history,
        question_history=question_history,
    )
