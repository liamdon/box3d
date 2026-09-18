// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "test_macros.h"

// b3GetVoxelFieldTriangle and b3RayCastAABB are internal
#include "aabb.h"
#include "shape.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

// Solid below floorHeight, empty above. Border voxels follow the same rule.
static b3VoxelFieldData* MakeFloorField( int countX, int countY, int countZ, int floorHeight, bool hasBorder )
{
	int count = countX * countY * countZ;
	uint8_t* voxels = calloc( count, 1 );
	for ( int z = 0; z < countZ; ++z )
	{
		for ( int y = 0; y < countY; ++y )
		{
			for ( int x = 0; x < countX; ++x )
			{
				voxels[x + countX * ( y + countY * z )] = y < floorHeight ? 1 : 0;
			}
		}
	}

	b3VoxelFieldDef def = { 0 };
	def.voxels = voxels;
	def.scale = b3Vec3_one;
	def.countX = countX;
	def.countY = countY;
	def.countZ = countZ;
	def.hasBorder = hasBorder;
	b3VoxelFieldData* field = b3CreateVoxelField( &def );
	free( voxels );
	return field;
}

static int VoxelFieldCreate( void )
{
	b3VoxelFieldData* field = MakeFloorField( 4, 4, 4, 2, false );

	ENSURE( field->version == B3_VOXEL_FIELD_VERSION );
	ENSURE( field->hash != 0 );
	ENSURE( field->countX == 4 && field->countY == 4 && field->countZ == 4 );
	ENSURE( field->solidCount == 32 );
	ENSURE( field->hasBorder == 0 );
	ENSURE( field->materialOffset == 0 );
	ENSURE( b3GetVoxelFieldMaterialIndices( field ) == NULL );
	ENSURE( field->byteCount > (int)sizeof( b3VoxelFieldData ) );

	ENSURE_SMALL( field->aabb.lowerBound.x, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.lowerBound.y, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.lowerBound.z, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.x - 4.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.y - 4.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.z - 4.0f, FLT_EPSILON );

	ENSURE( b3IsVoxelSolid( field, 0, 0, 0 ) == true );
	ENSURE( b3IsVoxelSolid( field, 3, 1, 3 ) == true );
	ENSURE( b3IsVoxelSolid( field, 0, 2, 0 ) == false );
	ENSURE( b3IsVoxelSolid( field, -1, 0, 0 ) == false );
	ENSURE( b3IsVoxelSolid( field, 0, 0, 4 ) == false );

	// Same input gives the same hash
	b3VoxelFieldData* twin = MakeFloorField( 4, 4, 4, 2, false );
	ENSURE( twin->hash == field->hash );
	ENSURE( twin->byteCount == field->byteCount );
	ENSURE( memcmp( twin, field, field->byteCount ) == 0 );

	b3DestroyVoxelField( twin );
	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelFieldMaterials( void )
{
	uint8_t voxels[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	uint8_t materials[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

	b3VoxelFieldDef def = { 0 };
	def.voxels = voxels;
	def.materialIndices = materials;
	def.scale = (b3Vec3){ 2.0f, 1.0f, 0.5f };
	def.countX = 2;
	def.countY = 2;
	def.countZ = 2;
	b3VoxelFieldData* field = b3CreateVoxelField( &def );

	ENSURE( field->materialOffset != 0 );
	const uint8_t* stored = b3GetVoxelFieldMaterialIndices( field );
	ENSURE( stored != NULL );
	ENSURE( memcmp( stored, materials, 8 ) == 0 );
	ENSURE( field->solidCount == 8 );
	ENSURE_SMALL( field->aabb.upperBound.x - 4.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.y - 2.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.z - 1.0f, FLT_EPSILON );

	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelFieldBorder( void )
{
	// 6x6x6 with a border: interior is 4x4x4, floor is 3 high including the border row
	b3VoxelFieldData* field = MakeFloorField( 6, 6, 6, 3, true );

	ENSURE( field->hasBorder == 1 );

	// Interior solid voxels: 4 * 4 * 2 (rows y = 1, 2 are interior; row 0 is the border)
	ENSURE( field->solidCount == 32 );

	// Bounds exclude the border
	ENSURE_SMALL( field->aabb.lowerBound.x - 1.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.lowerBound.y - 1.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.x - 5.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.z - 5.0f, FLT_EPSILON );

	// Border voxels are still readable as solid so they can hide faces
	ENSURE( b3IsVoxelSolid( field, 0, 0, 0 ) == true );

	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelFieldAABB( void )
{
	b3VoxelFieldData* field = MakeFloorField( 4, 2, 4, 1, false );
	b3Transform transform = { { 10.0f, 20.0f, 30.0f }, b3Quat_identity };
	b3AABB aabb = b3ComputeVoxelFieldAABB( field, transform );
	ENSURE_SMALL( aabb.lowerBound.x - 10.0f, FLT_EPSILON );
	ENSURE_SMALL( aabb.upperBound.y - 22.0f, FLT_EPSILON );
	ENSURE_SMALL( aabb.upperBound.z - 34.0f, FLT_EPSILON );
	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelWave( void )
{
	b3Vec3 scale = b3Vec3_one;
	b3VoxelFieldData* a = b3CreateVoxelWave( 10, 8, 10, 0, 0, scale, 0.3f, 0.5f, true );
	b3VoxelFieldData* b = b3CreateVoxelWave( 10, 8, 10, 8, 0, scale, 0.3f, 0.5f, true );

	// Field b starts 8 interior voxels to the +x of field a. Its left border column must equal
	// a's last interior column, and a's right border column must equal b's first interior column.
	for ( int z = 0; z < 10; ++z )
	{
		for ( int y = 0; y < 8; ++y )
		{
			ENSURE( b3IsVoxelSolid( b, 0, y, z ) == b3IsVoxelSolid( a, 8, y, z ) );
			ENSURE( b3IsVoxelSolid( a, 9, y, z ) == b3IsVoxelSolid( b, 1, y, z ) );
		}
	}

	ENSURE( a->solidCount > 0 );
	b3DestroyVoxelField( a );
	b3DestroyVoxelField( b );
	return 0;
}

// Map a face edge to the triangle that owns it and the edge bits within that triangle. Face edges 0
// and 1 belong to triangle 0 as edges 1 and 2. Face edges 2 and 3 belong to triangle 1 as edges 2 and 3.
static b3Triangle GetFaceEdgeTriangle( const b3VoxelFieldData* field, int voxelIndex, int face, int faceEdge, int* concaveBit,
									   int* inverseBit )
{
	static const int concaveBits[4] = { b3_concaveEdge1, b3_concaveEdge2, b3_concaveEdge2, b3_concaveEdge3 };
	static const int inverseBits[4] = { b3_inverseConcaveEdge1, b3_inverseConcaveEdge2, b3_inverseConcaveEdge2,
										b3_inverseConcaveEdge3 };
	int sub = faceEdge < 2 ? 0 : 1;
	*concaveBit = concaveBits[faceEdge];
	*inverseBit = inverseBits[faceEdge];
	return b3GetVoxelFieldTriangle( field, 12 * voxelIndex + 2 * face + sub );
}

// The corner and edge tables are the whole geometry. Check them against first principles.
static int VoxelFaceTables( void )
{
	// A single solid voxel: every face is exposed and every outer edge is convex.
	uint8_t voxel = 1;
	b3VoxelFieldDef def = { 0 };
	def.voxels = &voxel;
	def.scale = b3Vec3_one;
	def.countX = def.countY = def.countZ = 1;
	b3VoxelFieldData* field = b3CreateVoxelField( &def );

	static const float normals[6][3] = { { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };

	for ( int face = 0; face < 6; ++face )
	{
		for ( int sub = 0; sub < 2; ++sub )
		{
			b3Triangle t = b3GetVoxelFieldTriangle( field, 2 * face + sub );
			b3Vec3 n = b3MakeNormalFromPoints( t.vertices[0], t.vertices[1], t.vertices[2] );
			ENSURE_SMALL( n.x - normals[face][0], 1e-5f );
			ENSURE_SMALL( n.y - normals[face][1], 1e-5f );
			ENSURE_SMALL( n.z - normals[face][2], 1e-5f );

			// Every vertex lies on the face plane
			for ( int i = 0; i < 3; ++i )
			{
				float d = b3Dot( t.vertices[i], n );
				ENSURE_SMALL( d - b3MaxFloat( 0.0f, b3Dot( (b3Vec3){ 1.0f, 1.0f, 1.0f }, n ) ), 1e-5f );
			}

			// Both triangles share corner 0
			ENSURE( t.i1 == b3GetVoxelFieldTriangle( field, 2 * face ).i1 );
		}

		// Isolated voxel: outer edges convex, diagonal flat
		b3Triangle t0 = b3GetVoxelFieldTriangle( field, 2 * face );
		ENSURE( ( t0.flags & b3_concaveEdge1 ) == 0 && ( t0.flags & b3_inverseConcaveEdge1 ) != 0 );
		ENSURE( ( t0.flags & b3_concaveEdge2 ) == 0 && ( t0.flags & b3_inverseConcaveEdge2 ) != 0 );
		ENSURE( ( t0.flags & b3_flatEdge3 ) == b3_flatEdge3 );

		b3Triangle t1 = b3GetVoxelFieldTriangle( field, 2 * face + 1 );
		ENSURE( ( t1.flags & b3_flatEdge1 ) == b3_flatEdge1 );
		ENSURE( ( t1.flags & b3_concaveEdge2 ) == 0 && ( t1.flags & b3_inverseConcaveEdge2 ) != 0 );
		ENSURE( ( t1.flags & b3_concaveEdge3 ) == 0 && ( t1.flags & b3_inverseConcaveEdge3 ) != 0 );

		// The diagonal is shared: triangle 0 edge 3 is corners 2 to 0, triangle 1 edge 1 is corners 0 to 2
		ENSURE( t0.i3 == t1.i2 );
		ENSURE( t0.i1 == t1.i1 );
	}

	ENSURE( b3GetVoxelFieldTriangleCount( field ) == 12 );
	b3DestroyVoxelField( field );
	return 0;
}

// Derive the edge neighbor table from the geometry. Make the center voxel of a 3x3x3 field solid
// along with exactly one neighbor. The face edge that becomes flat must be the one whose two
// corners lie on the shared boundary with that neighbor.
static int VoxelEdgeNeighbors( void )
{
	static const int offsets[6][3] = { { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };

	for ( int n = 0; n < 6; ++n )
	{
		uint8_t voxels[27] = { 0 };
		int center = 1 + 3 * ( 1 + 3 * 1 );
		int neighbor = ( 1 + offsets[n][0] ) + 3 * ( ( 1 + offsets[n][1] ) + 3 * ( 1 + offsets[n][2] ) );
		voxels[center] = 1;
		voxels[neighbor] = 1;

		b3VoxelFieldDef def = { 0 };
		def.voxels = voxels;
		def.scale = b3Vec3_one;
		def.countX = def.countY = def.countZ = 3;
		b3VoxelFieldData* field = b3CreateVoxelField( &def );

		// The plane between the center and the neighbor
		int axis = n >> 1;
		float boundary = ( n & 1 ) ? 2.0f : 1.0f;

		for ( int face = 0; face < 6; ++face )
		{
			if ( face == n )
			{
				// Hidden face
				continue;
			}

			int flatEdgeCount = 0;
			for ( int edge = 0; edge < 4; ++edge )
			{
				int concaveBit, inverseBit;
				b3Triangle t = GetFaceEdgeTriangle( field, center, face, edge, &concaveBit, &inverseBit );
				bool concave = ( t.flags & concaveBit ) != 0;
				bool inverse = ( t.flags & inverseBit ) != 0;

				// The face edge corners are triangle vertices: face edges 0, 1 are triangle 0 vertices (0, 1) and (1, 2),
				// face edges 2, 3 are triangle 1 vertices (1, 2) and (2, 0).
				b3Vec3 c1, c2;
				if ( edge == 0 )
				{
					c1 = t.vertices[0], c2 = t.vertices[1];
				}
				else if ( edge == 1 || edge == 2 )
				{
					c1 = t.vertices[1], c2 = t.vertices[2];
				}
				else
				{
					c1 = t.vertices[2], c2 = t.vertices[0];
				}

				float v1[3] = { c1.x, c1.y, c1.z };
				float v2[3] = { c2.x, c2.y, c2.z };
				bool onBoundary = v1[axis] == boundary && v2[axis] == boundary;

				if ( onBoundary )
				{
					// Coplanar continuation onto the neighbor's face
					ENSURE( concave && inverse );
					flatEdgeCount += 1;
				}
				else
				{
					// Step down
					ENSURE( concave == false && inverse );
				}
			}

			// The neighbor touches every face along exactly one edge, except the face opposite the
			// neighbor, which shares nothing with it.
			int expectedFlatEdgeCount = face == ( n ^ 1 ) ? 0 : 1;
			ENSURE( flatEdgeCount == expectedFlatEdgeCount );
		}

		b3DestroyVoxelField( field );
	}

	return 0;
}

static int VoxelEdgeFlags( void )
{
	// 4x4x4, floor 1 high, plus a wall along x = 3 up to y = 3
	uint8_t voxels[64] = { 0 };
	for ( int z = 0; z < 4; ++z )
	{
		for ( int x = 0; x < 4; ++x )
		{
			voxels[x + 4 * ( 0 + 4 * z )] = 1;
			if ( x == 3 )
			{
				voxels[x + 4 * ( 1 + 4 * z )] = 1;
				voxels[x + 4 * ( 2 + 4 * z )] = 1;
			}
		}
	}

	b3VoxelFieldDef def = { 0 };
	def.voxels = voxels;
	def.scale = b3Vec3_one;
	def.countX = def.countY = def.countZ = 4;
	b3VoxelFieldData* field = b3CreateVoxelField( &def );

	// Top face (+y, face 3) of the floor voxel at (1, 0, 1): all four outer edges coplanar with
	// neighbors, so flat.
	{
		int index = 1 + 4 * ( 0 + 4 * 1 );
		b3Triangle t0 = b3GetVoxelFieldTriangle( field, 12 * index + 2 * 3 );
		b3Triangle t1 = b3GetVoxelFieldTriangle( field, 12 * index + 2 * 3 + 1 );
		ENSURE( ( t0.flags & b3_allFlatEdges ) == b3_allFlatEdges );
		ENSURE( ( t1.flags & b3_allFlatEdges ) == b3_allFlatEdges );
	}

	// Top face of (2, 0, 1): the +x edge meets the wall, so it is concave. Face 3 edge 2 is the +x
	// edge, which is triangle 1 edge 2.
	{
		int index = 2 + 4 * ( 0 + 4 * 1 );
		b3Triangle t1 = b3GetVoxelFieldTriangle( field, 12 * index + 2 * 3 + 1 );
		ENSURE( ( t1.flags & b3_concaveEdge2 ) != 0 );
		ENSURE( ( t1.flags & b3_inverseConcaveEdge2 ) == 0 );
	}

	// Top face of (0, 0, 1): the -x edge is the field boundary, so convex. Face 3 edge 0 is the
	// -x edge, which is triangle 0 edge 1.
	{
		int index = 0 + 4 * ( 0 + 4 * 1 );
		b3Triangle t0 = b3GetVoxelFieldTriangle( field, 12 * index + 2 * 3 );
		ENSURE( ( t0.flags & b3_concaveEdge1 ) == 0 );
		ENSURE( ( t0.flags & b3_inverseConcaveEdge1 ) != 0 );
	}

	// Adjacent top faces share grid corner indices
	{
		int a = 1 + 4 * ( 0 + 4 * 1 );
		int b = 2 + 4 * ( 0 + 4 * 1 );
		b3Triangle ta = b3GetVoxelFieldTriangle( field, 12 * a + 2 * 3 + 1 ); // corners 0, 2, 3 of a
		b3Triangle tb = b3GetVoxelFieldTriangle( field, 12 * b + 2 * 3 );	  // corners 0, 1, 2 of b

		// a's corner 3 is (x + 1, y + 1, z) and b's corner 0 is (x', y + 1, z) with x' = x + 1
		ENSURE( ta.i3 == tb.i1 );
		ENSURE( ta.vertices[2].x == tb.vertices[0].x );
		ENSURE( ta.vertices[2].y == tb.vertices[0].y );
		ENSURE( ta.vertices[2].z == tb.vertices[0].z );
	}

	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelMaterial( void )
{
	uint8_t voxels[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	uint8_t materials[8] = { 3, 1, 4, 1, 5, 9, 2, 6 };
	b3VoxelFieldDef def = { 0 };
	def.voxels = voxels;
	def.materialIndices = materials;
	def.scale = b3Vec3_one;
	def.countX = def.countY = def.countZ = 2;
	b3VoxelFieldData* field = b3CreateVoxelField( &def );

	for ( int v = 0; v < 8; ++v )
	{
		ENSURE( b3GetVoxelFieldMaterial( field, 12 * v + 7 ) == materials[v] );
	}

	b3DestroyVoxelField( field );

	def.materialIndices = NULL;
	field = b3CreateVoxelField( &def );
	ENSURE( b3GetVoxelFieldMaterial( field, 12 * 5 ) == 0 );
	b3DestroyVoxelField( field );
	return 0;
}

typedef struct QueryContext
{
	int indices[512];
	int count;
	int stopAt;
} QueryContext;

static bool CollectTriangles( b3Vec3 a, b3Vec3 b, b3Vec3 c, int triangleIndex, void* context )
{
	MAYBE_UNUSED( a );
	MAYBE_UNUSED( b );
	MAYBE_UNUSED( c );
	QueryContext* query = context;
	query->indices[query->count++] = triangleIndex;
	return query->count < query->stopAt;
}

static int VoxelQuerySorted( void )
{
	// 6x6x6 bordered field, floor 3 high: the interior floor is 4x4x2
	b3VoxelFieldData* field = MakeFloorField( 6, 6, 6, 3, true );

	QueryContext query = { 0 };
	query.stopAt = 512;
	b3AABB bounds = { { -10.0f, -10.0f, -10.0f }, { 10.0f, 10.0f, 10.0f } };
	b3QueryVoxelField( field, bounds, CollectTriangles, &query );

	// Only the 16 top faces are exposed. Every side face of an interior floor voxel touches a
	// solid interior neighbor or a solid border voxel, and every bottom face touches the solid
	// border row at y = 0.
	ENSURE( query.count == 2 * 16 );

	for ( int i = 1; i < query.count; ++i )
	{
		ENSURE( query.indices[i] > query.indices[i - 1] );
	}

	// Every reported triangle is a +y face (face 3)
	for ( int i = 0; i < query.count; ++i )
	{
		int face = ( query.indices[i] % 12 ) >> 1;
		ENSURE( face == 3 );
	}

	// A bounds that touches only one voxel column reports two triangles
	QueryContext one = { 0 };
	one.stopAt = 512;
	b3AABB small = { { 2.25f, 2.9f, 2.25f }, { 2.75f, 3.5f, 2.75f } };
	b3QueryVoxelField( field, small, CollectTriangles, &one );
	ENSURE( one.count == 2 );

	// Returning false stops the query
	QueryContext stop = { 0 };
	stop.stopAt = 3;
	b3QueryVoxelField( field, bounds, CollectTriangles, &stop );
	ENSURE( stop.count == 3 );

	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelRayCastFloor( void )
{
	b3VoxelFieldData* field = MakeFloorField( 8, 4, 8, 2, false );

	// Straight down onto the top of voxel (3, 1, 5)
	b3RayCastInput input = { { 3.5f, 10.0f, 5.5f }, { 0.0f, -20.0f, 0.0f }, 1.0f };
	b3CastOutput output = b3RayCastVoxelField( field, &input );
	ENSURE( output.hit );
	ENSURE_SMALL( output.fraction - 0.4f, 1e-5f );
	ENSURE_SMALL( output.point.y - 2.0f, 1e-4f );
	ENSURE_SMALL( output.normal.y - 1.0f, 1e-5f );
	int voxelIndex = output.triangleIndex / 12;
	int face = ( output.triangleIndex - 12 * voxelIndex ) >> 1;
	ENSURE( voxelIndex == 3 + 8 * ( 1 + 4 * 5 ) );
	ENSURE( face == 3 );
	ENSURE( output.materialIndex == 0 );

	// Sideways into the -x wall of the floor from outside the field
	input = (b3RayCastInput){ { -5.0f, 1.5f, 2.5f }, { 10.0f, 0.0f, 0.0f }, 1.0f };
	output = b3RayCastVoxelField( field, &input );
	ENSURE( output.hit );
	ENSURE_SMALL( output.fraction - 0.5f, 1e-5f );
	ENSURE_SMALL( output.normal.x + 1.0f, 1e-5f );

	// Miss: above the floor, horizontal
	input = (b3RayCastInput){ { -5.0f, 3.0f, 2.5f }, { 20.0f, 0.0f, 0.0f }, 1.0f };
	output = b3RayCastVoxelField( field, &input );
	ENSURE( output.hit == false );

	// Start inside the solid floor pointing up: the ray leaves through internal solid, then
	// exits into air. No exposed face faces the ray, so no hit.
	input = (b3RayCastInput){ { 3.5f, 0.5f, 5.5f }, { 0.0f, 10.0f, 0.0f }, 1.0f };
	output = b3RayCastVoxelField( field, &input );
	ENSURE( output.hit == false );

	// Max fraction limits the cast
	input = (b3RayCastInput){ { 3.5f, 10.0f, 5.5f }, { 0.0f, -20.0f, 0.0f }, 0.3f };
	output = b3RayCastVoxelField( field, &input );
	ENSURE( output.hit == false );

	b3DestroyVoxelField( field );
	return 0;
}

static float RandomFloat( uint32_t* state, float lower, float upper )
{
	*state = 1664525u * *state + 1013904223u;
	float u = (float)( *state >> 8 ) * ( 1.0f / 16777216.0f );
	return lower + ( upper - lower ) * u;
}

// Deterministic pseudo random field for brute force comparisons
static b3VoxelFieldData* MakeRandomField( int count, uint32_t seed, float fill )
{
	int total = count * count * count;
	uint8_t* voxels = calloc( total, 1 );
	uint32_t state = seed;
	for ( int i = 0; i < total; ++i )
	{
		voxels[i] = RandomFloat( &state, 0.0f, 1.0f ) < fill ? 1 : 0;
	}

	b3VoxelFieldDef def = { 0 };
	def.voxels = voxels;
	def.scale = (b3Vec3){ 1.0f, 0.5f, 1.5f };
	def.countX = def.countY = def.countZ = count;
	b3VoxelFieldData* field = b3CreateVoxelField( &def );
	free( voxels );
	return field;
}

static bool IsPointInSolid( const b3VoxelFieldData* field, b3Vec3 p )
{
	int x = (int)floorf( p.x / field->scale.x );
	int y = (int)floorf( p.y / field->scale.y );
	int z = (int)floorf( p.z / field->scale.z );
	return b3IsVoxelSolid( field, x, y, z );
}

static int VoxelRayCastBruteForce( void )
{
	b3VoxelFieldData* field = MakeRandomField( 8, 12345u, 0.3f );
	uint32_t state = 777u;
	int hitCount = 0;

	for ( int trial = 0; trial < 300; ++trial )
	{
		b3Vec3 origin = { RandomFloat( &state, -4.0f, 12.0f ), RandomFloat( &state, -2.0f, 6.0f ),
						  RandomFloat( &state, -6.0f, 18.0f ) };
		if ( IsPointInSolid( field, origin ) )
		{
			continue;
		}

		b3Vec3 translation = { RandomFloat( &state, -20.0f, 20.0f ), RandomFloat( &state, -10.0f, 10.0f ),
							   RandomFloat( &state, -30.0f, 30.0f ) };
		b3RayCastInput input = { origin, translation, 1.0f };
		b3CastOutput output = b3RayCastVoxelField( field, &input );

		// Brute force: the first solid voxel box hit along the ray. The origin is in air, so the
		// entry face of that box is exposed.
		float best = FLT_MAX;
		b3Vec3 p2 = b3Add( origin, translation );
		for ( int z = 0; z < 8; ++z )
		{
			for ( int y = 0; y < 8; ++y )
			{
				for ( int x = 0; x < 8; ++x )
				{
					if ( b3IsVoxelSolid( field, x, y, z ) == false )
					{
						continue;
					}

					b3AABB box = { b3Mul( field->scale, (b3Vec3){ (float)x, (float)y, (float)z } ),
								   b3Mul( field->scale, (b3Vec3){ (float)x + 1, (float)y + 1, (float)z + 1 } ) };
					float t1 = 0.0f, t2 = 1.0f;
					if ( b3RayCastAABB( box, origin, p2, &t1, &t2 ) && t1 < best )
					{
						best = t1;
					}
				}
			}
		}

		ENSURE( output.hit == ( best < FLT_MAX ) );
		if ( output.hit )
		{
			ENSURE_SMALL( output.fraction - best, 1e-4f );
			hitCount += 1;
		}
	}

	ENSURE( hitCount > 50 );
	b3DestroyVoxelField( field );
	return 0;
}

int VoxelFieldTest( void )
{
	RUN_SUBTEST( VoxelFieldCreate );
	RUN_SUBTEST( VoxelFieldMaterials );
	RUN_SUBTEST( VoxelFieldBorder );
	RUN_SUBTEST( VoxelFieldAABB );
	RUN_SUBTEST( VoxelWave );
	RUN_SUBTEST( VoxelFaceTables );
	RUN_SUBTEST( VoxelEdgeNeighbors );
	RUN_SUBTEST( VoxelEdgeFlags );
	RUN_SUBTEST( VoxelMaterial );
	RUN_SUBTEST( VoxelQuerySorted );
	RUN_SUBTEST( VoxelRayCastFloor );
	RUN_SUBTEST( VoxelRayCastBruteForce );

	return 0;
}
