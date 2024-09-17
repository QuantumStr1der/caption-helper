import OpenAI from "openai";
import sharp from 'sharp';

export const dynamic = "force-dynamic";

async function downscaleImage(base64Image: string, maxSize: number): Promise<string> {
  const buffer = Buffer.from(base64Image, 'base64');
  const image = sharp(buffer);
  const metadata = await image.metadata();

  if (metadata.width && metadata.height) {
    const longerAxis = Math.max(metadata.width, metadata.height);
    if (longerAxis > maxSize) {
      const resizedImage = await image
        .resize({
          width: metadata.width > metadata.height ? maxSize : undefined,
          height: metadata.height > metadata.width ? maxSize : undefined,
          fit: 'inside'
        })
        .toBuffer();
      return resizedImage.toString('base64');
    }
  }

  return base64Image;
}

export async function POST(req: Request) {
  try {
    const { image, apiKey, customToken, customInstruction, inherentAttributes, currentCaption } = await req.json();

    if (!apiKey) {
      throw new Error("OpenAI API key is missing");
    }

    const openai = new OpenAI({ apiKey });

    let systemPrompt = `
You are an AI assistant that captions images for training purposes. Your task is to create clear, detailed captions`;

    if (customToken) {
      systemPrompt += ` that always incorporate the custom token "${customToken}" at the beginning.`;
    }

    systemPrompt += `
The following guide outlines the captioning approach:

### Captioning Principles:
1. **Describe the Subject’s Pose and Expression**: Provide a detailed description of the subject’s body language and facial expression.
2. **Describe Appearance, Not Core Traits**: Exclude the main teaching concept but capture the subject’s features like clothing, hairstyle, and accessories.
3. **Incorporate Background and Lighting**:
   - Mention any environmental details that contribute to the scene’s mood or atmosphere.
   - Use specific descriptions to reflect the impact of lighting or surroundings.
4. **Limit to 77 Tokens, When Necessary**: You may use up to the full 77 tokens if it provides the most complete and clear description. It’s not necessary to use fewer if more tokens capture the image better.
5. **Follow Token Count Rule of Thumb**:
   - **1 token ≈ 0.75 words**.
   - **1 token ≈ 4 characters (including spaces)**.
   - Plan captions accordingly to avoid cutting off responses.
6. **Avoid Special Characters**: The response must not include any special characters except for periods (.) and commas (,). Ensure that no other punctuation or symbols are used in the output.

### Caption Structure:
1. **Globals**: Include rare tokens or consistent tags (e.g., character name, specific label).
1.5. **Natural Language Description**: Summarize the scene briefly (e.g., "Aerith gazes upward with wide eyes and a look of wonder").
2. **Pose and Perspective**:
   - Provide a general overview of the subject’s positioning and angle (e.g., "close-up," "profile view").
3. **Actions and State**:
   - Use action verbs to describe what the subject is doing (e.g., "gazing upward," "lips parted slightly").
4. **Subject Descriptions**:
   - Describe the subject's appearance in detail, excluding the main concept (e.g., "brown hair tied with a pink ribbon," "wears a red jacket over a white shirt").
5. **Notable Elements**:
   - Highlight specific details not part of the background (e.g., "the pink ribbon in her hair stands out").
6. **Background and Lighting**:
   - Describe the background or lighting that adds to the context (e.g., "warm, glowing lights in the background cast a soft glow on her face").
7. **Mood and Emotion**:
   - Capture the mood conveyed by the subject or scene (e.g., "a sense of awe or fascination").

Combine all of these to create a detailed caption for the image. If more words better convey the image, feel free to use the full 77 tokens. Only use fewer when appropriate. Ensure no special characters are used except periods and commas.
`;

    if (inherentAttributes) {
      systemPrompt += `
### Inherent Attributes to Avoid:
${inherentAttributes}
`;
    }

    if (customInstruction) {
      systemPrompt += `
${customInstruction}
`;
    }

    const downscaledImage = await downscaleImage(image, 1024);

    let userMessage = "Here is an image for you to describe. Please describe the image in detail and ensure it adheres to the guidelines. Do not include any uncertainty (i.e. I dont know, appears, seems) or any other text. Focus exclusively on visible elements and not conceptual ones.";

    if (currentCaption) {
      userMessage += ` The user says this about the image: "${currentCaption}". Consider this information while creating your caption, but don't simply repeat it. Provide your own detailed description.`;
    }

    userMessage += " Thank you very much for your help!";

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: systemPrompt,
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: userMessage,
            },
            {
              type: "image_url",
              image_url: {
                url: `data:image/jpeg;base64,${downscaledImage}`,
              },
            },
          ],
        },
      ],
      max_tokens: 300,
    });

    const caption = response.choices[0]?.message?.content || "";

    return new Response(JSON.stringify({ caption }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error processing request:", error);
    let errorMessage = "An unknown error occurred";
    let errorCode = "UNKNOWN_ERROR";

    if (error instanceof Error) {
      errorMessage = error.message;
      if ('code' in error) {
        errorCode = (error as any).code;
      }
    }

    return new Response(JSON.stringify({ error: errorMessage, code: errorCode }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
