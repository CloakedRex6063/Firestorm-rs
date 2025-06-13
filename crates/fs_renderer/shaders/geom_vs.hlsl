struct Vertex
{
    float3 Position;
    float UVx;
    float3 Normal;
    float UVy;
    float4 Tangent;
};

struct PerDrawConstants
{
    uint VertexBufferIndex;
};

ConstantBuffer<PerDrawConstants> PerDraw : register(b0);

void main(
        in  uint   inVertexIndex   : SV_VertexID,
        out float4 outPosition     : SV_Position,
        out float2 outUv           : TEXCOORD,
        out float4 outTangent      : TANGENT,
        out float4 outColor        : COLOR)
{
    StructuredBuffer<Vertex> vertexBuffer = ResourceDescriptorHeap[PerDraw.VertexBufferIndex];
    Vertex vertex = vertexBuffer.Load(inVertexIndex);

    outPosition = float4(vertex.Position, 1.0);
    outUv = float2(vertex.UVx, vertex.UVy);
    outColor = float4(1,1,1,1);   
}