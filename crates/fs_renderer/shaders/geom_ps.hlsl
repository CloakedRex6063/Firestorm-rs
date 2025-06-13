
struct PS_INPUT 
{
    float4 outPosition     : SV_Position;
    float2 outUV           : TEXCOORD;
    float4 outTangent      : TANGENT;
    float4 outColor        : COLOR;
};

float4 main(PS_INPUT input) : SV_Target
{
    return input.outColor;
}