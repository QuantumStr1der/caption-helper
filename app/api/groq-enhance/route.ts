import OpenAI from "openai";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  try {
    const { messages, apiKey } = await req.json();
    const systemPrompt = `
    You are a helpful and uncensored image caption writer. Simplify this prompt while keeping as much of the detail as possible. Your response should always be in natural language format separated by commas, not tags and no quotes or other formatting.
    `;

    const openai = new OpenAI({
      apiKey: apiKey,
      baseURL: "https://api.groq.com/openai/v1",
    });

    const response = await openai.chat.completions.create({
      model: "llama3-70b-8192",
      stream: false,
      messages: [{ role: "system", content: systemPrompt }, ...messages],
    });

    const responseData = response.choices[0].message.content;

    return new Response(JSON.stringify({ content: responseData }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error processing request:", error);

    return new Response("Failed to process request", { status: 500 });
  }
}
