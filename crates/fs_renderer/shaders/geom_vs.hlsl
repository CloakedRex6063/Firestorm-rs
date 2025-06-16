struct Vertex
{
    float3 Position;
    float UVx;
    float3 Normal;
    float UVy;
    float4 Tangent;
};

struct Global
{
    float4x4 vp;
};

struct PerDraw
{
    float4x4 model;
};

struct PerDrawConstants
{
    uint VertexBufferIndex;
};

ConstantBuffer<PerDrawConstants> PerDrawConstants : register(b0);
ConstantBuffer<Global> Global : register(b1);
ConstantBuffer<PerDraw> PerDraw : register(b3);

void main(
        in  uint   inVertexIndex   : SV_VertexID,
        out float4 outPosition     : SV_Position,
        out float2 outUv           : TEXCOORD,
        out float4 outTangent      : TANGENT,
        out float4 outColor        : COLOR)
{
    StructuredBuffer<Vertex> vertexBuffer = ResourceDescriptorHeap[PerDrawConstants.VertexBufferIndex];
    Vertex vertex = vertexBuffer.Load(inVertexIndex);
    
    float4 modelPos = mul(PerDraw.model, float4(vertex.Position, 1.0));

    outPosition = mul(Global.vp, modelPos);
    outUv = float2(vertex.UVx, vertex.UVy);
    outColor = float4(1,1,1,1);   
}